
#ifdef __ve__
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
//#define VE_SETERRQ
#endif

/*
  Code for interpolating between grids represented by DMDAs
*/

/*
      For linear elements there are two branches of code to compute the interpolation. They should compute the same results but may not. The "new version" does
   not work for periodic domains, the old does. Change NEWVERSION to 1 to compile in the new version. Eventually when we are sure the two produce identical results
   we will remove/merge the new version. Based on current tests, these both produce the same results. We are leaving NEWVERSION for now in the code since some
   consider it cleaner, but old version is turned on since it handles periodic case.
*/
#define NEWVERSION 0

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

/*
   Since the interpolation uses MATMAIJ for dof > 0 we convert request for non-MATAIJ baseded matrices to MATAIJ.
   This is a bit of a hack, the reason for it is partially because -dm_mat_type defines the
   matrix type for both the operator matrices and the interpolation matrices so that users
   can select matrix types of base MATAIJ for accelerators
*/
static PetscErrorCode ConvertToAIJ(MatType intype,MatType *outtype)
{
  PetscErrorCode ierr;
  PetscInt       i;
  char           const *types[3] = {MATAIJ,MATSEQAIJ,MATMPIAIJ};
  PetscBool      flg;

  PetscFunctionBegin;
  *outtype = MATAIJ;
  for (i=0; i<3; i++) {
    ierr = PetscStrbeginswith(intype,types[i],&flg);CHKERRQ(ierr);
    if (flg) {
      *outtype = intype;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInterpolation_DA_1D_Q1(DM dac,DM daf,Mat *A)
{
  PetscErrorCode         ierr;
  PetscInt               i,i_start,m_f,Mx;
  const PetscInt         *idx_f,*idx_c;
  PetscInt               m_ghost,m_ghost_c;
  PetscInt               row,col,i_start_ghost,mx,m_c,nc,ratio;
  PetscInt               i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar            v[2],x;
  Mat                    mat;
  DMBoundaryType         bx;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  MatType                mattype;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&bx,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (bx == DM_BOUNDARY_PERIODIC) {
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }

  ierr = DMDAGetCorners(daf,&i_start,NULL,NULL,&m_f,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,NULL,NULL,&m_ghost,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,NULL,NULL,&m_c,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,NULL,NULL,&m_ghost_c,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)dac),&mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /*
     Temporary hack: Since the MAIJ matrix must be converted to AIJ before being used by the GPU
     we don't want the original unconverted matrix copied to the GPU
   */
  if (dof > 1) {
    ierr = MatBindToCPU(mat,PETSC_TRUE);CHKERRQ(ierr);
  }
  #endif
  ierr = MatSetSizes(mat,m_f,m_c,mx,Mx);CHKERRQ(ierr);
  ierr = ConvertToAIJ(dac->mattype,&mattype);CHKERRQ(ierr);
  ierr = MatSetType(mat,mattype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,2,NULL,1,NULL);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  if (!NEWVERSION) {

    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row = idx_f[i-i_start_ghost];

      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /*
       Only include those interpolation points that are truly
       nonzero. Note this is very important for final grid lines
       in x direction; since they have no right neighbor
       */
      x  = ((PetscReal)(i - i_c*ratio))/((PetscReal)ratio);
      nc = 0;
      /* one left and below; or we are right on it */
      col      = (i_c-i_start_ghost_c);
      cols[nc] = idx_c[col];
      v[nc++]  = -x + 1.0;
      /* one right? */
      if (i_c*ratio != i) {
        cols[nc] = idx_c[col+1];
        v[nc++]  = x;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
    }

  } else {
    PetscScalar *xi;
    PetscInt    li,nxi,n;
    PetscScalar Ni[2];

    /* compute local coordinate arrays */
    nxi  = ratio + 1;
    ierr = PetscMalloc1(nxi,&xi);CHKERRQ(ierr);
    for (li=0; li<nxi; li++) {
      xi[li] = -1.0 + (PetscScalar)li*(2.0/(PetscScalar)(nxi-1));
    }

    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row = idx_f[(i-i_start_ghost)];

      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /* remainders */
      li = i - ratio * (i/ratio);
      if (i==mx-1) li = nxi-1;

      /* corners */
      col     = (i_c-i_start_ghost_c);
      cols[0] = idx_c[col];
      Ni[0]   = 1.0;
      if ((li==0) || (li==nxi-1)) {
        ierr = MatSetValue(mat,row,cols[0],Ni[0],INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }

      /* edges + interior */
      /* remainders */
      if (i==mx-1) i_c--;

      col     = (i_c-i_start_ghost_c);
      cols[0] = idx_c[col]; /* one left and below; or we are right on it */
      cols[1] = idx_c[col+1];

      Ni[0] = 0.5*(1.0-xi[li]);
      Ni[1] = 0.5*(1.0+xi[li]);
      for (n=0; n<2; n++) {
        if (PetscAbsScalar(Ni[n])<1.0e-32) cols[n]=-1;
      }
      ierr = MatSetValues(mat,1,&row,2,cols,Ni,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(xi);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInterpolation_DA_1D_Q0(DM dac,DM daf,Mat *A)
{
  PetscErrorCode         ierr;
  PetscInt               i,i_start,m_f,Mx;
  const PetscInt         *idx_f,*idx_c;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  PetscInt               m_ghost,m_ghost_c;
  PetscInt               row,col,i_start_ghost,mx,m_c,nc,ratio;
  PetscInt               i_c,i_start_c,i_start_ghost_c,cols[2],dof;
  PetscScalar            v[2],x;
  Mat                    mat;
  DMBoundaryType         bx;
  MatType                mattype;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&bx,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (bx == DM_BOUNDARY_PERIODIC) {
    if (!Mx) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be positive",Mx);
    ratio = mx/Mx;
    if (ratio*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    if (Mx < 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be greater than 1",Mx);
    ratio = (mx-1)/(Mx-1);
    if (ratio*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }

  ierr = DMDAGetCorners(daf,&i_start,NULL,NULL,&m_f,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,NULL,NULL,&m_ghost,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,NULL,NULL,&m_c,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,NULL,NULL,&m_ghost_c,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)dac),&mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /*
     Temporary hack: Since the MAIJ matrix must be converted to AIJ before being used by the GPU
     we don't want the original unconverted matrix copied to the GPU
   */
  if (dof > 1) {
    ierr = MatBindToCPU(mat,PETSC_TRUE);CHKERRQ(ierr);
  }
  #endif
  ierr = MatSetSizes(mat,m_f,m_c,mx,Mx);CHKERRQ(ierr);
  ierr = ConvertToAIJ(dac->mattype,&mattype);CHKERRQ(ierr);
  ierr = MatSetType(mat,mattype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,2,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,2,NULL,0,NULL);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (i=i_start; i<i_start+m_f; i++) {
    /* convert to local "natural" numbering and then to PETSc global numbering */
    row = idx_f[(i-i_start_ghost)];

    i_c = (i/ratio);    /* coarse grid node to left of fine grid node */

    /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x direction; since they have no right neighbor
    */
    x  = ((PetscReal)(i - i_c*ratio))/((PetscReal)ratio);
    nc = 0;
    /* one left and below; or we are right on it */
    col      = (i_c-i_start_ghost_c);
    cols[nc] = idx_c[col];
    v[nc++]  = -x + 1.0;
    /* one right? */
    if (i_c*ratio != i) {
      cols[nc] = idx_c[col+1];
      v[nc++]  = x;
    }
    ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(5.0*m_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef __ve__
#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv,orow,ocol)     \
{ \
    if (col <= lastcol1)  low1 = 0;     \
    else                 high1 = nrow1; \
    lastcol1 = col;\
    while (high1-low1 > 5) { \
      t = (low1+high1)/2; \
      if (rp1[t] > col) high1 = t; \
      else              low1  = t; \
    } \
      for (_i=low1; _i<high1; _i++) { \
        if (rp1[_i] > col) break; \
        if (rp1[_i] == col) { \
          if (addv == ADD_VALUES) { \
            ap1[_i] += value;   \
            /* Not sure LogFlops will slow dow the code or not */ \
            (void)PetscLogFlops(1.0);   \
           } \
          else                    ap1[_i] = value; \
          inserted = PETSC_TRUE; \
          goto a_noinsert; \
        } \
      }  \
      if (value == 0.0 && ignorezeroentries && row != col) {low1 = 0; high1 = nrow1;goto a_noinsert;} \
      if (nonew == 1) {low1 = 0; high1 = nrow1; goto a_noinsert;}                \
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
      MatSeqXAIJReallocateAIJ(A,am,1,nrow1,row,col,rmax1,aa,ai,aj,rp1,ap1,aimax,nonew,MatScalar); \
      N = nrow1++ - 1; a->nz++; high1++; \
      /* shift up all the later entries in this row */ \
      ierr = PetscArraymove(rp1+_i+1,rp1+_i,N-_i+1);CHKERRQ(ierr);\
      ierr = PetscArraymove(ap1+_i+1,ap1+_i,N-_i+1);CHKERRQ(ierr);\
      rp1[_i] = col;  \
      ap1[_i] = value;  \
      A->nonzerostate++;\
      a_noinsert: ; \
      ailen[row] = nrow1; \
}

#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv,orow,ocol) \
  { \
    if (col <= lastcol2) low2 = 0;                        \
    else high2 = nrow2;                                   \
    lastcol2 = col;                                       \
    while (high2-low2 > 5) {                              \
      t = (low2+high2)/2;                                 \
      if (rp2[t] > col) high2 = t;                        \
      else             low2  = t;                         \
    }                                                     \
    for (_i=low2; _i<high2; _i++) {                       \
      if (rp2[_i] > col) break;                           \
      if (rp2[_i] == col) {                               \
        if (addv == ADD_VALUES) {                         \
          ap2[_i] += value;                               \
          (void)PetscLogFlops(1.0);                       \
        }                                                 \
        else                    ap2[_i] = value;          \
        inserted = PETSC_TRUE;                            \
        goto b_noinsert;                                  \
      }                                                   \
    }                                                     \
    if (value == 0.0 && ignorezeroentries) {low2 = 0; high2 = nrow2; goto b_noinsert;} \
    if (nonew == 1) {low2 = 0; high2 = nrow2; goto b_noinsert;}                        \
    if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B,bm,1,nrow2,row,col,rmax2,ba,bi,bj,rp2,ap2,bimax,nonew,MatScalar); \
    N = nrow2++ - 1; b->nz++; high2++;                    \
    /* shift up all the later entries in this row */      \
    ierr = PetscArraymove(rp2+_i+1,rp2+_i,N-_i+1);CHKERRQ(ierr);\
    ierr = PetscArraymove(ap2+_i+1,ap2+_i,N-_i+1);CHKERRQ(ierr);\
    rp2[_i] = col;                                        \
    ap2[_i] = value;                                      \
    B->nonzerostate++;                                    \
    b_noinsert: ;                                         \
    bilen[row] = nrow2;                                   \
  }
#endif

PetscErrorCode DMCreateInterpolation_DA_2D_Q1(DM dac,DM daf,Mat *A)
{
  PetscErrorCode         ierr;
  PetscInt               i,j,i_start,j_start,m_f,n_f,Mx,My,dof;
  const PetscInt         *idx_c,*idx_f;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  PetscInt               m_ghost,n_ghost,m_ghost_c,n_ghost_c,*dnz,*onz;
#ifdef __ve__
  PetscInt               row,col,i_start_ghost,j_start_ghost,mx,m_c,my,nc,ratioi,ratioj;
#else
  PetscInt               row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratioi,ratioj;
#endif
  PetscInt               i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c,col_shift,col_scale;
  PetscMPIInt            size_c,size_f,rank_f;
#ifdef __ve__
  PetscScalar            x,y;
#else
  PetscScalar            v[4],x,y;
#endif
  Mat                    mat;
  DMBoundaryType         bx,by;
  MatType                mattype;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,&My,NULL,NULL,NULL,NULL,NULL,NULL,&bx,&by,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,&my,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (bx == DM_BOUNDARY_PERIODIC) {
    if (!Mx) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be positive",Mx);
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    if (Mx < 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be greater than 1",Mx);
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (by == DM_BOUNDARY_PERIODIC) {
    if (!My) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of y coarse grid points %D must be positive",My);
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    if (My < 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of y coarse grid points %D must be greater than 1",My);
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }

  ierr = DMDAGetCorners(daf,&i_start,&j_start,NULL,&m_f,&n_f,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,NULL,&m_ghost,&n_ghost,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,NULL,&m_c,&n_c,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,NULL,&m_ghost_c,&n_ghost_c,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

#ifdef __ve__
  PetscInt cols[4*(i_start+m_f)], nc_tb[4*(i_start+m_f)], row_tb[4*(i_start+m_f)];
  PetscScalar v[4*(i_start+m_f)];
#endif

  /*
   Used for handling a coarse DMDA that lives on 1/4 the processors of the fine DMDA.
   The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the
   processors). It's effective length is hence 4 times its normal length, this is
   why the col_scale is multiplied by the interpolation matrix column sizes.
   sol_shift allows each set of 1/4 processors do its own interpolation using ITS
   copy of the coarse vector. A bit of a hack but you do better.

   In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
   */
  ierr      = MPI_Comm_size(PetscObjectComm((PetscObject)dac),&size_c);CHKERRMPI(ierr);
  ierr      = MPI_Comm_size(PetscObjectComm((PetscObject)daf),&size_f);CHKERRMPI(ierr);
  ierr      = MPI_Comm_rank(PetscObjectComm((PetscObject)daf),&rank_f);CHKERRMPI(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*(rank_f/size_c);

  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)daf),m_f*n_f,col_scale*m_c*n_c,dnz,onz);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                          i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /*
       Only include those interpolation points that are truly
       nonzero. Note this is very important for final grid lines
       in x and y directions; since they have no right/top neighbors
       */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = col_shift + idx_c[col];
      /* one right and below */
      if (i_c*ratioi != i) cols[nc++] = col_shift + idx_c[col+1];
      /* one left and above */
      if (j_c*ratioj != j) cols[nc++] = col_shift + idx_c[col+m_ghost_c];
      /* one right and above */
      if (i_c*ratioi != i && j_c*ratioj != j) cols[nc++] = col_shift + idx_c[col+(m_ghost_c+1)];
      ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatCreate(PetscObjectComm((PetscObject)daf),&mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /*
     Temporary hack: Since the MAIJ matrix must be converted to AIJ before being used by the GPU
     we don't want the original unconverted matrix copied to the GPU
  */
  if (dof > 1) {
    ierr = MatBindToCPU(mat,PETSC_TRUE);CHKERRQ(ierr);
  }
#endif
  ierr = MatSetSizes(mat,m_f*n_f,col_scale*m_c*n_c,mx*my,col_scale*Mx*My);CHKERRQ(ierr);
  ierr = ConvertToAIJ(dac->mattype,&mattype);CHKERRQ(ierr);
  ierr = MatSetType(mat,mattype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  if (!NEWVERSION) {

#ifdef __ve__
   PetscBool flg, flg2;
   ierr = MatSetValues_IsMPIAIJ(mat,&flg);
   ierr = MatSetValues_IsSeqAIJ(mat,&flg2);
   if (flg==PETSC_TRUE && !PetscDefined(USE_DEBUG) && !mat->was_assembled) {
    for (j=j_start; j<j_start+n_f; j++) {
     PetscInt jj;
     for (jj=0;jj<4;jj++) {
#pragma _NEC ivdep
      for (i=i_start; i<i_start+m_f; i++) {
       if (jj==0) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row_tb[i] = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */

        /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
        x = ((PetscReal)(i - i_c*ratioi))/((PetscReal)ratioi);
        y = ((PetscReal)(j - j_c*ratioj))/((PetscReal)ratioj);

        nc_tb[i] = 0;
        /* one left and below; or we are right on it */
        col      = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col];
        v[(nc_tb[i]++)*(i_start+m_f)+i]  = x*y - x - y + 1.0;
        /* one right and below */
        if (i_c*ratioi != i) {
          cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col+1];
          v[(nc_tb[i]++)*(i_start+m_f)+i]  = -x*y + x;
        }
        /* one left and above */
        if (j_c*ratioj != j) {
          cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col+m_ghost_c];
          v[(nc_tb[i]++)*(i_start+m_f)+i]  = -x*y + y;
        }
        /* one right and above */
        if (j_c*ratioj != j && i_c*ratioi != i) {
          cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col+(m_ghost_c+1)];
          v[(nc_tb[i]++)*(i_start+m_f)+i]  = x*y;
        }
       }
        //ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
        //PetscErrorCode MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
        //-- enter --
        {
          PetscErrorCode ierr;
          PetscInt m;
          const PetscInt *idxm;
          PetscInt n;
          const PetscInt *idxn;
          InsertMode addv;
        
          m = 1;
          idxm = row_tb;
          n = nc_tb[i];
          idxn = cols;
          addv = INSERT_VALUES;
        
          PetscFunctionBeginHot;
          PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
          PetscValidType(mat,1);
          if (!m || !n) continue; /* no values to insert */
          PetscValidIntPointer(idxm,3);
          PetscValidIntPointer(idxn,5);
          MatCheckPreallocated(mat,1);
        
          //ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
          //PetscErrorCode MatSetValues_MPIAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
          //-- enter --
          {
            Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
            PetscScalar    value = 0.0;
            PetscErrorCode ierr;
            PetscInt       rstart  = mat->rmap->rstart,rend = mat->rmap->rend;
            PetscInt       cstart      = mat->cmap->rstart,cend = mat->cmap->rend,row,col;
            PetscBool      roworiented = aij->roworiented;
          
            const PetscInt *im;
            const PetscInt *in;
          
            im = idxm;
            in = idxn;
          
            /* Some Variables required in the macro */
            Mat        A                    = aij->A;
            Mat_SeqAIJ *a                   = (Mat_SeqAIJ*)A->data;
            PetscInt   *aimax               = a->imax,*ai = a->i,*ailen = a->ilen,*aj = a->j;
            PetscBool  ignorezeroentries    = a->ignorezeroentries;
            Mat        B                    = aij->B;
            Mat_SeqAIJ *b                   = (Mat_SeqAIJ*)B->data;
            PetscInt   *bimax               = b->imax,*bi = b->i,*bilen = b->ilen,*bj = b->j,bm = aij->B->rmap->n,am = aij->A->rmap->n;
            MatScalar  *aa,*ba;
            /* This variable below is only for the PETSC_HAVE_VIENNACL or PETSC_HAVE_CUDA cases, but we define it in all cases because we
             * cannot use "#if defined" inside a macro. */
            PETSC_UNUSED PetscBool inserted = PETSC_FALSE;
          
            PetscInt  *rp1,*rp2,ii,nrow1,nrow2,_i,rmax1,rmax2,N,low1,high1,low2,high2,t,lastcol1,lastcol2;
            PetscInt  nonew;
            MatScalar *ap1,*ap2;
          
            PetscFunctionBegin;
#if defined(PETSC_HAVE_DEVICE)
            if (A->offloadmask == PETSC_OFFLOAD_GPU) {
              const PetscScalar *dummy;
              ierr = MatSeqAIJGetArrayRead(A,&dummy);CHKERRQ(ierr);
              ierr = MatSeqAIJRestoreArrayRead(A,&dummy);CHKERRQ(ierr);
            }
            if (B->offloadmask == PETSC_OFFLOAD_GPU) {
              const PetscScalar *dummy;
              ierr = MatSeqAIJGetArrayRead(B,&dummy);CHKERRQ(ierr);
              ierr = MatSeqAIJRestoreArrayRead(B,&dummy);CHKERRQ(ierr);
            }
#endif
            aa = a->a;
            ba = b->a;
            ii=0;
            if (ii<m) { // max(m)=max(nrow)=1
              if (im[ii*(i_start+m_f)+i] < 0) /* continue */ ; else
              {
#ifdef VE_SETERRQ
              if (PetscUnlikely(im[ii*(i_start+m_f)+i] >= mat->rmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[ii*(i_start+m_f)+i],mat->rmap->N-1);
#endif
              if (im[ii*(i_start+m_f)+i] >= rstart && im[ii*(i_start+m_f)+i] < rend) {
                row      = im[ii*(i_start+m_f)+i] - rstart;
                lastcol1 = -1;
                rp1      = aj + ai[row];
                ap1      = aa + ai[row];
                rmax1    = aimax[row];
                nrow1    = ailen[row];
                low1     = 0;
                high1    = nrow1;
                lastcol2 = -1;
                rp2      = bj + bi[row];
                ap2      = ba + bi[row];
                rmax2    = bimax[row];
                nrow2    = bilen[row];
                low2     = 0;
                high2    = nrow2;
          
                if (jj<n) { // max(n)=max(ncol)=1
                  if (v)  value = roworiented ? v[(ii*n+jj)*(i_start+m_f)+i] : v[(ii+jj*m)*(i_start+m_f)+i];
                  if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES) && im[ii*(i_start+m_f)+i] != in[jj*(i_start+m_f)+i]) {
                    ; // continue;
                  } else
                  if (in[jj*(i_start+m_f)+i] >= cstart && in[jj*(i_start+m_f)+i] < cend) {
                    col   = in[jj*(i_start+m_f)+i] - cstart;
                    nonew = a->nonew;

                    //MatSetValues_SeqAIJ_A_Private(row,col,value,addv,im[ii*(i_start+m_f)+i],in[jj*(i_start+m_f)+i]);
                    //#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv,orow,ocol)
                    { //-- enter
                      PetscInt  orow,ocol;
                      PetscBool bflg, gflg; // bflg means break flag, gflg means goto flag
                      orow = im[ii*(i_start+m_f)+i];
                      ocol = in[jj*(i_start+m_f)+i];
                      if (col <= lastcol1)  low1 = 0;
                      else                 high1 = nrow1;
                      lastcol1 = col;
                      if (high1-low1 > 5) { // max count=1
                        t = (low1+high1)/2;
                        if (rp1[t] > col) high1 = t;
                        else              low1  = t;
                      }
                      gflg=PETSC_FALSE;
                      _i=low1;
                      bflg=PETSC_FALSE;
                      if (_i<high1 && bflg==PETSC_FALSE) { // max(high1-low1+1)=1
                        if (rp1[_i] > col) bflg = PETSC_TRUE;
                        if (rp1[_i] == col) {
                          if (addv == ADD_VALUES) {
                            ap1[_i] += value;
                            /* Not sure LogFlops will slow dow the code or not */
                            //(void)PetscLogFlops(1.0);
                           }
                          else                    ap1[_i] = value;
                          inserted = PETSC_TRUE;
                          bflg = gflg = PETSC_TRUE;
                        }
                        if (bflg==PETSC_FALSE) _i++;
                      }
                      if (_i<high1 && bflg==PETSC_FALSE) { // max(high1-low1+1)=2
                        if (rp1[_i] > col) bflg = PETSC_TRUE;
                        if (rp1[_i] == col) {
                          if (addv == ADD_VALUES) {
                            ap1[_i] += value;
                            /* Not sure LogFlops will slow dow the code or not */
                            //(void)PetscLogFlops(1.0);
                           }
                          else                    ap1[_i] = value;
                          inserted = PETSC_TRUE;
                          bflg = gflg = PETSC_TRUE;
                        }
                        if (bflg==PETSC_FALSE) _i++;
                      }
                      if (_i<high1 && bflg==PETSC_FALSE) { // max(high1-low1+1)=3
                        if (rp1[_i] > col) bflg = PETSC_TRUE;
                        if (rp1[_i] == col) {
                          if (addv == ADD_VALUES) {
                            ap1[_i] += value;
                            /* Not sure LogFlops will slow dow the code or not */
                            //(void)PetscLogFlops(1.0);
                           }
                          else                    ap1[_i] = value;
                          inserted = PETSC_TRUE;
                          bflg = gflg = PETSC_TRUE;
                        }
                        if (bflg==PETSC_FALSE) _i++;
                      }
                      if (_i<high1 && bflg==PETSC_FALSE) { // max(high1-low1+1)=4
                        if (rp1[_i] > col) bflg = PETSC_TRUE;
                        if (rp1[_i] == col) {
                          if (addv == ADD_VALUES) {
                            ap1[_i] += value;
                            /* Not sure LogFlops will slow dow the code or not */
                            //(void)PetscLogFlops(1.0);
                           }
                          else                    ap1[_i] = value;
                          inserted = PETSC_TRUE;
                          bflg = gflg = PETSC_TRUE;
                        }
                        if (bflg==PETSC_FALSE) _i++;
                      }
                      if (gflg == PETSC_FALSE && value == 0.0 && ignorezeroentries && row != col) {low1 = 0; high1 = nrow1;gflg = PETSC_TRUE;}
                      if (gflg == PETSC_FALSE && nonew == 1) {low1 = 0; high1 = nrow1; gflg = PETSC_TRUE;}
                      if (gflg==PETSC_FALSE) {
#ifdef VE_SETERRQ
                        if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol);
#endif
                        // MatSeqXAIJReallocateAIJ(A,am,1,nrow1,row,col,rmax1,aa,ai,aj,rp1,ap1,aimax,nonew,MatScalar);
                        // if nrow1<=rmax1 do nothing
                        N = nrow1++ - 1; a->nz++; high1++;
                        /* shift up all the later entries in this row */
                        //ierr = PetscArraymove(rp1+_i+1,rp1+_i,N-_i+1);CHKERRQ(ierr);
                        //ierr = PetscArraymove(ap1+_i+1,ap1+_i,N-_i+1);CHKERRQ(ierr);
                        rp1[_i] = col;
                        ap1[_i] = value;
                        A->nonzerostate++;
                      }
                      //a_noinsert: ;
                      ailen[row] = nrow1;
                    } //-- exit --
#if defined(PETSC_HAVE_DEVICE)
                    if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
                  } else if (in[jj*(i_start+m_f)+i] < 0) {;} // continue
#ifdef VE_SETERRQ
                  else if (in[jj*(i_start+m_f)+i] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[jj*(i_start+m_f)+i],mat->cmap->N-1);
#endif
                  else {
                    col = in[jj*(i_start+m_f)+i];
                    nonew = b->nonew;
                    //MatSetValues_SeqAIJ_B_Private(row,col,value,addv,im[ii*(i_start+m_f)+i],in[jj*(i_start+m_f)+i]);
                    //#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv,orow,ocol)
                    { //-- enter --
                      PetscInt  orow,ocol;
                      PetscBool bflg, gflg; // bflg means break flag, gflg means goto flag
                      orow = im[ii*(i_start+m_f)+i];
                      ocol = in[jj*(i_start+m_f)+i];

                      if (col <= lastcol2) low2 = 0;
                      else high2 = nrow2;
                      lastcol2 = col;
                      if (high2-low2 > 5) { // max count=1
                        t = (low2+high2)/2;
                        if (rp2[t] > col) high2 = t;
                        else             low2  = t;
                      }
                      gflg = PETSC_FALSE;
                      _i=low2;
                      bflg = PETSC_FALSE;
                      if (_i<high2 && bflg==PETSC_FALSE) { // max(high2-low2+1)=1
                        if (rp2[_i] > col) bflg = PETSC_TRUE;
                        if (rp2[_i] == col) {
                          if (addv == ADD_VALUES) {
                            ap2[_i] += value;
                            //(void)PetscLogFlops(1.0);
                          }
                          else                    ap2[_i] = value;
                          inserted = PETSC_TRUE;
                          bflg = gflg = PETSC_FALSE;
                        }
                        if (bflg==PETSC_FALSE) _i++;
                      }
                      if (_i<high2 && bflg==PETSC_FALSE) { // max(high2-low2+1)=2
                        if (rp2[_i] > col) bflg = PETSC_TRUE;
                        if (rp2[_i] == col) {
                          if (addv == ADD_VALUES) {
                            ap2[_i] += value;
                            //(void)PetscLogFlops(1.0);
                          }
                          else                    ap2[_i] = value;
                          inserted = PETSC_TRUE;
                          bflg = gflg = PETSC_FALSE;
                        }
                        if (bflg==PETSC_FALSE) _i++;
                      }
                      if (_i<high2 && bflg==PETSC_FALSE) { // max(high2-low2+1)=3
                        if (rp2[_i] > col) bflg = PETSC_TRUE;
                        if (rp2[_i] == col) {
                          if (addv == ADD_VALUES) {
                            ap2[_i] += value;
                            //(void)PetscLogFlops(1.0);
                          }
                          else                    ap2[_i] = value;
                          inserted = PETSC_TRUE;
                          bflg = gflg = PETSC_FALSE;
                        }
                        if (bflg==PETSC_FALSE) _i++;
                      }
                      if (gflg == PETSC_FALSE && value == 0.0 && ignorezeroentries) {low2 = 0; high2 = nrow2;gflg = PETSC_TRUE;}
                      if (gflg == PETSC_FALSE && nonew == 1) {low2 = 0; high2 = nrow2; gflg = PETSC_TRUE;}
                      if (gflg == PETSC_FALSE ) {
#ifdef VE_SETERRQ
                        if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol);
#endif
                        // MatSeqXAIJReallocateAIJ(B,bm,1,nrow2,row,col,rmax2,ba,bi,bj,rp2,ap2,bimax,nonew,MatScalar);
                        // if nrow2<=rmax2 do nothing
                        N = nrow2++ - 1; b->nz++; high2++;
                        /* shift up all the later entries in this row */
                        //ierr = PetscArraymove(rp2+_i+1,rp2+_i,N-_i+1);CHKERRQ(ierr);
                        //ierr = PetscArraymove(ap2+_i+1,ap2+_i,N-_i+1);CHKERRQ(ierr);
                        rp2[_i] = col;
                        ap2[_i] = value;
                        B->nonzerostate++;
                      }
                      //b_noinsert: ;
                      bilen[row] = nrow2;
                    } //-- exit --
#if defined(PETSC_HAVE_DEVICE)
                    if (B->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) B->offloadmask = PETSC_OFFLOAD_CPU;
#endif
                  }
                }
              } else {
#ifdef VE_SETERRQ
                if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i*(i_start+m_f)+j]);
#endif
              }
              }
              ii++;
            }
            ierr = 0;
            //PetscFunctionReturn(0);
          }
          //-- exit --
          //CHKERRQ(ierr);
          ierr = 0;
          //PetscFunctionReturn(0);
        }
        ierr = 0;CHKERRQ(ierr);
        //-- exit --
      }
     }
    }
   } else if (flg2==PETSC_TRUE && !PetscDefined(USE_DEBUG) && !mat->was_assembled) {
    for (j=j_start; j<j_start+n_f; j++) {
      PetscInt k,l;
     for (l=0; l<5; l++) {
#pragma _NEC ivdep
      for (i=i_start; i<i_start+m_f; i++) {
       if (l==0) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row_tb[i] = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */

        /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
        x = ((PetscReal)(i - i_c*ratioi))/((PetscReal)ratioi);
        y = ((PetscReal)(j - j_c*ratioj))/((PetscReal)ratioj);

        nc_tb[i] = 0;
        /* one left and below; or we are right on it */
        col      = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col];
        v[(nc_tb[i]++)*(i_start+m_f)+i]  = x*y - x - y + 1.0;
        /* one right and below */
        if (i_c*ratioi != i) {
          cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col+1];
          v[(nc_tb[i]++)*(i_start+m_f)+i]  = -x*y + x;
        }
        /* one left and above */
        if (j_c*ratioj != j) {
          cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col+m_ghost_c];
          v[(nc_tb[i]++)*(i_start+m_f)+i]  = -x*y + y;
        }
        /* one right and above */
        if (j_c*ratioj != j && i_c*ratioi != i) {
          cols[nc_tb[i]*(i_start+m_f)+i] = col_shift + idx_c[col+(m_ghost_c+1)];
          v[(nc_tb[i]++)*(i_start+m_f)+i]  = x*y;
        }
       }
        //ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
        //PetscErrorCode MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
        //-- enter --
        {
          PetscErrorCode ierr;
          PetscInt m;
          const PetscInt *idxm;
          PetscInt n;
          const PetscInt *idxn;
          InsertMode addv;
        
          m = 1;
          idxm = row_tb;
          n = nc_tb[i];
          idxn = cols;
          addv = INSERT_VALUES;
        
          PetscFunctionBeginHot;
          PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
          PetscValidType(mat,1);
          if (!m || !n) continue; /* no values to insert */
          PetscValidIntPointer(idxm,3);
          PetscValidIntPointer(idxn,5);
          MatCheckPreallocated(mat,1);
        
          //ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
          //PetscErrorCode MatSetValues_SeqAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
          //-- enter --
          {
            Mat A;
            A = mat;
            Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
            PetscInt       *rp,low,high,t,ii,row,nrow,col,rmax,N;
            PetscInt       *imax = a->imax,*ai = a->i,*ailen = a->ilen;
            PetscErrorCode ierr;
            PetscInt       *aj = a->j,nonew = a->nonew,lastcol = -1;
            MatScalar      *ap=NULL,value=0.0,*aa;
            PetscBool      ignorezeroentries = a->ignorezeroentries;
            PetscBool      roworiented       = a->roworiented;
#if defined(PETSC_HAVE_DEVICE)
            PetscBool      inserted          = PETSC_FALSE;
#endif
            const PetscInt *im;
            const PetscInt *in;
            InsertMode is;

            im = idxm;
            in = idxn;
            is = addv;

            PetscFunctionBegin;
#if defined(PETSC_HAVE_DEVICE)
            if (A->offloadmask == PETSC_OFFLOAD_GPU) {
              const PetscScalar *dummy;
              ierr = MatSeqAIJGetArrayRead(A,&dummy);CHKERRQ(ierr);
              ierr = MatSeqAIJRestoreArrayRead(A,&dummy);CHKERRQ(ierr);
            }
#endif
            aa = a->a;
            k=0;
            if (k<m) { /* loop over added rows */ // max(m)=1
              row = im[k*(i_start+m_f)+i];
             if (row >= 0) { // if row<0 continue;
#ifdef VE_SETERRQ
              if (PetscUnlikelyDebug(row >= A->rmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->n-1);
#endif
              rp   = aj + ai[row];
              if (!A->structure_only) ap = aa + ai[row];
              rmax = imax[row]; nrow = ailen[row];
              low  = 0;
              high = nrow;
              if (l<n) { /* loop over added columns */ // max(n)=1
               PetscBool bflg, gflg; // bflg means break flag, gflg means goto flag
               if (in[l*(i_start+m_f)+i] >= 0) { // if in[] < 0 continue;
#ifdef VE_SETERRQ
                if (PetscUnlikelyDebug(in[l*(i_start+m_f)+i] >= A->cmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l*(i_start+m_f)+i],A->cmap->n-1);
#endif
                col = in[l*(i_start+m_f)+i];
                if (v && !A->structure_only) value = roworiented ? v[(l + k*n)*(i_start+m_f)+i] : v[(k + l*m)*(i_start+m_f)+i];
                if (!A->structure_only && value == 0.0 && ignorezeroentries && is == ADD_VALUES && row != col) continue;

                if (col <= lastcol) low = 0;
                else high = nrow;
                lastcol = col;
                if (high-low > 5) { // while num=1
                  t = (low+high)/2;
                  if (rp[t] > col) high = t;
                  else low = t;
                }
                gflg=PETSC_FALSE;
                ii=low;
                bflg=PETSC_FALSE;
                if (ii<high && bflg==PETSC_FALSE) { // max(high-low)=1
                  if (rp[ii] > col) bflg = PETSC_TRUE;
                  if (rp[ii] == col) {
                    if (!A->structure_only) {
                      if (is == ADD_VALUES) {
                        ap[ii] += value;
                        //(void)PetscLogFlops(1.0);
                      }
                      else ap[ii] = value;
#if defined(PETSC_HAVE_DEVICE)
                      inserted = PETSC_TRUE;
#endif
                    }
                    low = ii + 1;
                    bflg = gflg = PETSC_TRUE;
                  }
                  ii++;
                }
                if (ii<high && bflg==PETSC_FALSE) { // max(high-low)=2
                  if (rp[ii] > col) bflg = PETSC_TRUE;
                  if (rp[ii] == col) {
                    if (!A->structure_only) {
                      if (is == ADD_VALUES) {
                        ap[ii] += value;
                        //(void)PetscLogFlops(1.0);
                      }
                      else ap[ii] = value;
#if defined(PETSC_HAVE_DEVICE)
                      inserted = PETSC_TRUE;
#endif
                    }
                    low = ii + 1;
                    bflg = gflg = PETSC_TRUE;
                  }
                  ii++;
                }
                if (ii<high && bflg==PETSC_FALSE) { // max(high-low)=3
                  if (rp[ii] > col) bflg = PETSC_TRUE;
                  if (rp[ii] == col) {
                    if (!A->structure_only) {
                      if (is == ADD_VALUES) {
                        ap[ii] += value;
                        //(void)PetscLogFlops(1.0);
                      }
                      else ap[ii] = value;
#if defined(PETSC_HAVE_DEVICE)
                      inserted = PETSC_TRUE;
#endif
                    }
                    low = ii + 1;
                    bflg = gflg = PETSC_TRUE;
                  }
                  ii++;
                }
                if (ii<high && bflg==PETSC_FALSE) { // max(high-low)=4
                  if (rp[ii] > col) bflg = PETSC_TRUE;
                  if (rp[ii] == col) {
                    if (!A->structure_only) {
                      if (is == ADD_VALUES) {
                        ap[ii] += value;
                        //(void)PetscLogFlops(1.0);
                      }
                      else ap[ii] = value;
#if defined(PETSC_HAVE_DEVICE)
                      inserted = PETSC_TRUE;
#endif
                    }
                    low = ii + 1;
                    bflg = gflg = PETSC_TRUE;
                  }
                  ii++;
                }
                if (ii<high && bflg==PETSC_FALSE) { // max(high-low)=5
                  if (rp[ii] > col) bflg = PETSC_TRUE;
                  if (rp[ii] == col) {
                    if (!A->structure_only) {
                      if (is == ADD_VALUES) {
                        ap[ii] += value;
                        //(void)PetscLogFlops(1.0);
                      }
                      else ap[ii] = value;
#if defined(PETSC_HAVE_DEVICE)
                      inserted = PETSC_TRUE;
#endif
                    }
                    low = ii + 1;
                    bflg = gflg = PETSC_TRUE;
                  }
                  ii++;
                }
                if (value == 0.0 && ignorezeroentries && row != col) gflg = PETSC_TRUE;
                if (nonew == 1) gflg = PETSC_TRUE;
               if (gflg==PETSC_FALSE) {
#ifdef VE_SETERRQ
                if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at (%D,%D) in the matrix",row,col);
#endif
                if (A->structure_only) {
                  //MatSeqXAIJReallocateAIJ_structure_only(A,A->rmap->n,1,nrow,row,col,rmax,ai,aj,rp,imax,nonew,MatScalar);
                  // if nrow<rmax do nothing
                } else {
                  //MatSeqXAIJReallocateAIJ(A,A->rmap->n,1,nrow,row,col,rmax,aa,ai,aj,rp,ap,imax,nonew,MatScalar);
                  // if nrow<rmax do nothing
                }
                N = nrow++ - 1; a->nz++; high++;
                /* shift up all the later entries in this row */
                //ierr  = PetscArraymove(rp+ii+1,rp+ii,N-ii+1);CHKERRQ(ierr);
                // if N-ii+1>0 do nothing
                rp[ii] = col;
                if (!A->structure_only) {
                  //ierr  = PetscArraymove(ap+ii+1,ap+ii,N-ii+1);CHKERRQ(ierr);
                  // if N-ii+1>0 do nothing
                  ap[ii] = value;
                }
                low = ii + 1;
                A->nonzerostate++;
#if defined(PETSC_HAVE_DEVICE)
                inserted = PETSC_TRUE;
#endif
               }
//noinsert:;
               }
              }
              ailen[row] = nrow;
             }
              k++;
            }
#if defined(PETSC_HAVE_DEVICE)
            if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
            //PetscFunctionReturn(0);
          }
          ierr = 0;
          //-- exit --
          //PetscFunctionReturn(0);
        }
        ierr = 0; CHKERRQ(ierr);
        //-- exit --
      } // i-loop
     } // l-loop
    } // j-loop
   } else {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */

        /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
        x = ((PetscReal)(i - i_c*ratioi))/((PetscReal)ratioi);
        y = ((PetscReal)(j - j_c*ratioj))/((PetscReal)ratioj);

        nc = 0;
        /* one left and below; or we are right on it */
        col      = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc] = col_shift + idx_c[col];
        v[nc++]  = x*y - x - y + 1.0;
        /* one right and below */
        if (i_c*ratioi != i) {
          cols[nc] = col_shift + idx_c[col+1];
          v[nc++]  = -x*y + x;
        }
        /* one left and above */
        if (j_c*ratioj != j) {
          cols[nc] = col_shift + idx_c[col+m_ghost_c];
          v[nc++]  = -x*y + y;
        }
        /* one right and above */
        if (j_c*ratioj != j && i_c*ratioi != i) {
          cols[nc] = col_shift + idx_c[col+(m_ghost_c+1)];
          v[nc++]  = x*y;
        }
        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
   }
#else
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */

        /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
        x = ((PetscReal)(i - i_c*ratioi))/((PetscReal)ratioi);
        y = ((PetscReal)(j - j_c*ratioj))/((PetscReal)ratioj);

        nc = 0;
        /* one left and below; or we are right on it */
        col      = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc] = col_shift + idx_c[col];
        v[nc++]  = x*y - x - y + 1.0;
        /* one right and below */
        if (i_c*ratioi != i) {
          cols[nc] = col_shift + idx_c[col+1];
          v[nc++]  = -x*y + x;
        }
        /* one left and above */
        if (j_c*ratioj != j) {
          cols[nc] = col_shift + idx_c[col+m_ghost_c];
          v[nc++]  = -x*y + y;
        }
        /* one right and above */
        if (j_c*ratioj != j && i_c*ratioi != i) {
          cols[nc] = col_shift + idx_c[col+(m_ghost_c+1)];
          v[nc++]  = x*y;
        }
        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
#endif

  } else {
    PetscScalar Ni[4];
    PetscScalar *xi,*eta;
    PetscInt    li,nxi,lj,neta;

    /* compute local coordinate arrays */
    nxi  = ratioi + 1;
    neta = ratioj + 1;
    ierr = PetscMalloc1(nxi,&xi);CHKERRQ(ierr);
    ierr = PetscMalloc1(neta,&eta);CHKERRQ(ierr);
    for (li=0; li<nxi; li++) {
      xi[li] = -1.0 + (PetscScalar)li*(2.0/(PetscScalar)(nxi-1));
    }
    for (lj=0; lj<neta; lj++) {
      eta[lj] = -1.0 + (PetscScalar)lj*(2.0/(PetscScalar)(neta-1));
    }

    /* loop over local fine grid nodes setting interpolation for those*/
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */

        /* remainders */
        li = i - ratioi * (i/ratioi);
        if (i==mx-1) li = nxi-1;
        lj = j - ratioj * (j/ratioj);
        if (j==my-1) lj = neta-1;

        /* corners */
        col     = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[0] = col_shift + idx_c[col]; /* left, below */
        Ni[0]   = 1.0;
        if ((li==0) || (li==nxi-1)) {
          if ((lj==0) || (lj==neta-1)) {
            ierr = MatSetValue(mat,row,cols[0],Ni[0],INSERT_VALUES);CHKERRQ(ierr);
            continue;
          }
        }

        /* edges + interior */
        /* remainders */
        if (i==mx-1) i_c--;
        if (j==my-1) j_c--;

        col     = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[0] = col_shift + idx_c[col]; /* left, below */
        cols[1] = col_shift + idx_c[col+1]; /* right, below */
        cols[2] = col_shift + idx_c[col+m_ghost_c]; /* left, above */
        cols[3] = col_shift + idx_c[col+(m_ghost_c+1)]; /* right, above */

        Ni[0] = 0.25*(1.0-xi[li])*(1.0-eta[lj]);
        Ni[1] = 0.25*(1.0+xi[li])*(1.0-eta[lj]);
        Ni[2] = 0.25*(1.0-xi[li])*(1.0+eta[lj]);
        Ni[3] = 0.25*(1.0+xi[li])*(1.0+eta[lj]);

        nc = 0;
        if (PetscAbsScalar(Ni[0])<1.0e-32) cols[0]=-1;
        if (PetscAbsScalar(Ni[1])<1.0e-32) cols[1]=-1;
        if (PetscAbsScalar(Ni[2])<1.0e-32) cols[2]=-1;
        if (PetscAbsScalar(Ni[3])<1.0e-32) cols[3]=-1;

        ierr = MatSetValues(mat,1,&row,4,cols,Ni,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(xi);CHKERRQ(ierr);
    ierr = PetscFree(eta);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       Contributed by Andrei Draganescu <aidraga@sandia.gov>
*/
PetscErrorCode DMCreateInterpolation_DA_2D_Q0(DM dac,DM daf,Mat *A)
{
  PetscErrorCode         ierr;
  PetscInt               i,j,i_start,j_start,m_f,n_f,Mx,My,dof;
  const PetscInt         *idx_c,*idx_f;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  PetscInt               m_ghost,n_ghost,m_ghost_c,n_ghost_c,*dnz,*onz;
  PetscInt               row,col,i_start_ghost,j_start_ghost,cols[4],mx,m_c,my,nc,ratioi,ratioj;
  PetscInt               i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c,col_shift,col_scale;
  PetscMPIInt            size_c,size_f,rank_f;
  PetscScalar            v[4];
  Mat                    mat;
  DMBoundaryType         bx,by;
  MatType                mattype;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,&My,NULL,NULL,NULL,NULL,NULL,NULL,&bx,&by,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,&my,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (!Mx) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be positive",Mx);
  if (!My) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of y coarse grid points %D must be positive",My);
  ratioi = mx/Mx;
  ratioj = my/My;
  if (ratioi*Mx != mx) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in x");
  if (ratioj*My != my) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in y");
  if (ratioi != 2) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Coarsening factor in x must be 2");
  if (ratioj != 2) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Coarsening factor in y must be 2");

  ierr = DMDAGetCorners(daf,&i_start,&j_start,NULL,&m_f,&n_f,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,NULL,&m_ghost,&n_ghost,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,NULL,&m_c,&n_c,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,NULL,&m_ghost_c,&n_ghost_c,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  /*
     Used for handling a coarse DMDA that lives on 1/4 the processors of the fine DMDA.
     The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the
     processors). It's effective length is hence 4 times its normal length, this is
     why the col_scale is multiplied by the interpolation matrix column sizes.
     sol_shift allows each set of 1/4 processors do its own interpolation using ITS
     copy of the coarse vector. A bit of a hack but you do better.

     In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
  */
  ierr      = MPI_Comm_size(PetscObjectComm((PetscObject)dac),&size_c);CHKERRMPI(ierr);
  ierr      = MPI_Comm_size(PetscObjectComm((PetscObject)daf),&size_f);CHKERRMPI(ierr);
  ierr      = MPI_Comm_rank(PetscObjectComm((PetscObject)daf),&rank_f);CHKERRMPI(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*(rank_f/size_c);

  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)daf),m_f*n_f,col_scale*m_c*n_c,dnz,onz);CHKERRQ(ierr);
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */

      if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
      if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

      /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      nc = 0;
      /* one left and below; or we are right on it */
      col        = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc++] = col_shift + idx_c[col];
      ierr       = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  ierr = MatCreate(PetscObjectComm((PetscObject)daf),&mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /*
     Temporary hack: Since the MAIJ matrix must be converted to AIJ before being used by the GPU
     we don't want the original unconverted matrix copied to the GPU
  */
  if (dof > 1) {
    ierr = MatBindToCPU(mat,PETSC_TRUE);CHKERRQ(ierr);
  }
  #endif
  ierr = MatSetSizes(mat,m_f*n_f,col_scale*m_c*n_c,mx*my,col_scale*Mx*My);CHKERRQ(ierr);
  ierr = ConvertToAIJ(dac->mattype,&mattype);CHKERRQ(ierr);
  ierr = MatSetType(mat,mattype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (j=j_start; j<j_start+n_f; j++) {
    for (i=i_start; i<i_start+m_f; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row = idx_f[(m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

      i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratioj);    /* coarse grid node below fine grid node */
      nc  = 0;
      /* one left and below; or we are right on it */
      col      = (m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
      cols[nc] = col_shift + idx_c[col];
      v[nc++]  = 1.0;

      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(13.0*m_f*n_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       Contributed by Jianming Yang <jianming-yang@uiowa.edu>
*/
PetscErrorCode DMCreateInterpolation_DA_3D_Q0(DM dac,DM daf,Mat *A)
{
  PetscErrorCode         ierr;
  PetscInt               i,j,l,i_start,j_start,l_start,m_f,n_f,p_f,Mx,My,Mz,dof;
  const PetscInt         *idx_c,*idx_f;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  PetscInt               m_ghost,n_ghost,p_ghost,m_ghost_c,n_ghost_c,p_ghost_c,nc,*dnz,*onz;
  PetscInt               row,col,i_start_ghost,j_start_ghost,l_start_ghost,cols[8],mx,m_c,my,n_c,mz,p_c,ratioi,ratioj,ratiol;
  PetscInt               i_c,j_c,l_c,i_start_c,j_start_c,l_start_c,i_start_ghost_c,j_start_ghost_c,l_start_ghost_c,col_shift,col_scale;
  PetscMPIInt            size_c,size_f,rank_f;
  PetscScalar            v[8];
  Mat                    mat;
  DMBoundaryType         bx,by,bz;
  MatType                mattype;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,&My,&Mz,NULL,NULL,NULL,NULL,NULL,&bx,&by,&bz,NULL);CHKERRQ(ierr);
  if (!Mx) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be positive",Mx);
  if (!My) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of y coarse grid points %D must be positive",My);
  if (!Mz) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of z coarse grid points %D must be positive",Mz);
  ierr = DMDAGetInfo(daf,NULL,&mx,&my,&mz,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ratioi = mx/Mx;
  ratioj = my/My;
  ratiol = mz/Mz;
  if (ratioi*Mx != mx) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in x");
  if (ratioj*My != my) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in y");
  if (ratiol*Mz != mz) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Fine grid points must be multiple of coarse grid points in z");
  if (ratioi != 2 && ratioi != 1) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Coarsening factor in x must be 1 or 2");
  if (ratioj != 2 && ratioj != 1) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Coarsening factor in y must be 1 or 2");
  if (ratiol != 2 && ratiol != 1) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_WRONG,"Coarsening factor in z must be 1 or 2");

  ierr = DMDAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  /*
     Used for handling a coarse DMDA that lives on 1/4 the processors of the fine DMDA.
     The coarse vector is then duplicated 4 times (each time it lives on 1/4 of the
     processors). It's effective length is hence 4 times its normal length, this is
     why the col_scale is multiplied by the interpolation matrix column sizes.
     sol_shift allows each set of 1/4 processors do its own interpolation using ITS
     copy of the coarse vector. A bit of a hack but you do better.

     In the standard case when size_f == size_c col_scale == 1 and col_shift == 0
  */
  ierr      = MPI_Comm_size(PetscObjectComm((PetscObject)dac),&size_c);CHKERRMPI(ierr);
  ierr      = MPI_Comm_size(PetscObjectComm((PetscObject)daf),&size_f);CHKERRMPI(ierr);
  ierr      = MPI_Comm_rank(PetscObjectComm((PetscObject)daf),&rank_f);CHKERRMPI(ierr);
  col_scale = size_f/size_c;
  col_shift = Mx*My*Mz*(rank_f/size_c);

  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)daf),m_f*n_f*p_f,col_scale*m_c*n_c*p_c,dnz,onz);CHKERRQ(ierr);
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */
        l_c = (l/ratiol);

        if (l_c < l_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    l_start %D l_c %D l_start_ghost_c %D",l_start,l_c,l_start_ghost_c);
        if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
        if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

        /*
           Only include those interpolation points that are truly
           nonzero. Note this is very important for final grid lines
           in x and y directions; since they have no right/top neighbors
        */
        nc = 0;
        /* one left and below; or we are right on it */
        col        = (m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc++] = col_shift + idx_c[col];
        ierr       = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatCreate(PetscObjectComm((PetscObject)daf),&mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /*
     Temporary hack: Since the MAIJ matrix must be converted to AIJ before being used by the GPU
     we don't want the original unconverted matrix copied to the GPU
  */
  if (dof > 1) {
    ierr = MatBindToCPU(mat,PETSC_TRUE);CHKERRQ(ierr);
  }
  #endif
  ierr = MatSetSizes(mat,m_f*n_f*p_f,col_scale*m_c*n_c*p_c,mx*my*mz,col_scale*Mx*My*Mz);CHKERRQ(ierr);
  ierr = ConvertToAIJ(dac->mattype,&mattype);CHKERRQ(ierr);
  ierr = MatSetType(mat,mattype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

        i_c = (i/ratioi);    /* coarse grid node to left of fine grid node */
        j_c = (j/ratioj);    /* coarse grid node below fine grid node */
        l_c = (l/ratiol);
        nc  = 0;
        /* one left and below; or we are right on it */
        col      = (m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc] = col_shift + idx_c[col];
        v[nc++]  = 1.0;

        ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscLogFlops(13.0*m_f*n_f*p_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInterpolation_DA_3D_Q1(DM dac,DM daf,Mat *A)
{
  PetscErrorCode         ierr;
  PetscInt               i,j,i_start,j_start,m_f,n_f,Mx,My,dof,l;
  const PetscInt         *idx_c,*idx_f;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  PetscInt               m_ghost,n_ghost,m_ghost_c,n_ghost_c,Mz,mz;
  PetscInt               row,col,i_start_ghost,j_start_ghost,cols[8],mx,m_c,my,nc,ratioi,ratioj,ratiok;
  PetscInt               i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  PetscInt               l_start,p_f,l_start_ghost,p_ghost,l_start_c,p_c;
  PetscInt               l_start_ghost_c,p_ghost_c,l_c,*dnz,*onz;
  PetscScalar            v[8],x,y,z;
  Mat                    mat;
  DMBoundaryType         bx,by,bz;
  MatType                mattype;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,&My,&Mz,NULL,NULL,NULL,NULL,NULL,&bx,&by,&bz,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,&my,&mz,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (mx == Mx) {
    ratioi = 1;
  } else if (bx == DM_BOUNDARY_PERIODIC) {
    if (!Mx) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be positive",Mx);
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    if (Mx < 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of x coarse grid points %D must be greater than 1",Mx);
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (my == My) {
    ratioj = 1;
  } else if (by == DM_BOUNDARY_PERIODIC) {
    if (!My) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of y coarse grid points %D must be positive",My);
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    if (My < 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of y coarse grid points %D must be greater than 1",My);
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }
  if (mz == Mz) {
    ratiok = 1;
  } else if (bz == DM_BOUNDARY_PERIODIC) {
    if (!Mz) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of z coarse grid points %D must be positive",Mz);
    ratiok = mz/Mz;
    if (ratiok*Mz != mz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mz/Mz  must be integer: mz %D Mz %D",mz,Mz);
  } else {
    if (Mz < 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of z coarse grid points %D must be greater than 1",Mz);
    ratiok = (mz-1)/(Mz-1);
    if (ratiok*(Mz-1) != mz-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mz - 1)/(Mz - 1) must be integer: mz %D Mz %D",mz,Mz);
  }

  ierr = DMDAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix, determining exact preallocation */
  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)dac),m_f*n_f*p_f,m_c*n_c*p_c,dnz,onz);CHKERRQ(ierr);
  /* loop over local fine grid nodes counting interpolating points */
  for (l=l_start; l<l_start+p_f; l++) {
    for (j=j_start; j<j_start+n_f; j++) {
      for (i=i_start; i<i_start+m_f; i++) {
        /* convert to local "natural" numbering and then to PETSc global numbering */
        row = idx_f[(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];
        i_c = (i/ratioi);
        j_c = (j/ratioj);
        l_c = (l/ratiok);
        if (l_c < l_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                            l_start %D l_c %D l_start_ghost_c %D",l_start,l_c,l_start_ghost_c);
        if (j_c < j_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                            j_start %D j_c %D j_start_ghost_c %D",j_start,j_c,j_start_ghost_c);
        if (i_c < i_start_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
                                            i_start %D i_c %D i_start_ghost_c %D",i_start,i_c,i_start_ghost_c);

        /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
         */
        nc         = 0;
        col        = (m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
        cols[nc++] = idx_c[col];
        if (i_c*ratioi != i) {
          cols[nc++] = idx_c[col+1];
        }
        if (j_c*ratioj != j) {
          cols[nc++] = idx_c[col+m_ghost_c];
        }
        if (l_c*ratiok != l) {
          cols[nc++] = idx_c[col+m_ghost_c*n_ghost_c];
        }
        if (j_c*ratioj != j && i_c*ratioi != i) {
          cols[nc++] = idx_c[col+(m_ghost_c+1)];
        }
        if (j_c*ratioj != j && l_c*ratiok != l) {
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)];
        }
        if (i_c*ratioi != i && l_c*ratiok != l) {
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+1)];
        }
        if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) {
          cols[nc++] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)];
        }
        ierr = MatPreallocateSet(row,nc,cols,dnz,onz);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatCreate(PetscObjectComm((PetscObject)dac),&mat);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  /*
     Temporary hack: Since the MAIJ matrix must be converted to AIJ before being used by the GPU
     we don't want the original unconverted matrix copied to the GPU
  */
  if (dof > 1) {
    ierr = MatBindToCPU(mat,PETSC_TRUE);CHKERRQ(ierr);
  }
  #endif
  ierr = MatSetSizes(mat,m_f*n_f*p_f,m_c*n_c*p_c,mx*my*mz,Mx*My*Mz);CHKERRQ(ierr);
  ierr = ConvertToAIJ(dac->mattype,&mattype);CHKERRQ(ierr);
  ierr = MatSetType(mat,mattype);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(mat,0,dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(mat,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  if (!NEWVERSION) {

    for (l=l_start; l<l_start+p_f; l++) {
      for (j=j_start; j<j_start+n_f; j++) {
        for (i=i_start; i<i_start+m_f; i++) {
          /* convert to local "natural" numbering and then to PETSc global numbering */
          row = idx_f[(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

          i_c = (i/ratioi);
          j_c = (j/ratioj);
          l_c = (l/ratiok);

          /*
           Only include those interpolation points that are truly
           nonzero. Note this is very important for final grid lines
           in x and y directions; since they have no right/top neighbors
           */
          x = ((PetscReal)(i - i_c*ratioi))/((PetscReal)ratioi);
          y = ((PetscReal)(j - j_c*ratioj))/((PetscReal)ratioj);
          z = ((PetscReal)(l - l_c*ratiok))/((PetscReal)ratiok);

          nc = 0;
          /* one left and below; or we are right on it */
          col = (m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c)+m_ghost_c*(j_c-j_start_ghost_c)+(i_c-i_start_ghost_c));

          cols[nc] = idx_c[col];
          v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));

          if (i_c*ratioi != i) {
            cols[nc] = idx_c[col+1];
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. - (2.0*z-1.));
          }

          if (j_c*ratioj != j) {
            cols[nc] = idx_c[col+m_ghost_c];
            v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
          }

          if (l_c*ratiok != l) {
            cols[nc] = idx_c[col+m_ghost_c*n_ghost_c];
            v[nc++]  = .125*(1. - (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
          }

          if (j_c*ratioj != j && i_c*ratioi != i) {
            cols[nc] = idx_c[col+(m_ghost_c+1)];
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. - (2.0*z-1.));
          }

          if (j_c*ratioj != j && l_c*ratiok != l) {
            cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)];
            v[nc++]  = .125*(1. - (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
          }

          if (i_c*ratioi != i && l_c*ratiok != l) {
            cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+1)];
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. - (2.0*y-1.))*(1. + (2.0*z-1.));
          }

          if (i_c*ratioi != i && l_c*ratiok != l && j_c*ratioj != j) {
            cols[nc] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)];
            v[nc++]  = .125*(1. + (2.0*x-1.))*(1. + (2.0*y-1.))*(1. + (2.0*z-1.));
          }
          ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }

  } else {
    PetscScalar *xi,*eta,*zeta;
    PetscInt    li,nxi,lj,neta,lk,nzeta,n;
    PetscScalar Ni[8];

    /* compute local coordinate arrays */
    nxi   = ratioi + 1;
    neta  = ratioj + 1;
    nzeta = ratiok + 1;
    ierr  = PetscMalloc1(nxi,&xi);CHKERRQ(ierr);
    ierr  = PetscMalloc1(neta,&eta);CHKERRQ(ierr);
    ierr  = PetscMalloc1(nzeta,&zeta);CHKERRQ(ierr);
    for (li=0; li<nxi; li++) xi[li] = -1.0 + (PetscScalar)li*(2.0/(PetscScalar)(nxi-1));
    for (lj=0; lj<neta; lj++) eta[lj] = -1.0 + (PetscScalar)lj*(2.0/(PetscScalar)(neta-1));
    for (lk=0; lk<nzeta; lk++) zeta[lk] = -1.0 + (PetscScalar)lk*(2.0/(PetscScalar)(nzeta-1));

    for (l=l_start; l<l_start+p_f; l++) {
      for (j=j_start; j<j_start+n_f; j++) {
        for (i=i_start; i<i_start+m_f; i++) {
          /* convert to local "natural" numbering and then to PETSc global numbering */
          row = idx_f[(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))];

          i_c = (i/ratioi);
          j_c = (j/ratioj);
          l_c = (l/ratiok);

          /* remainders */
          li = i - ratioi * (i/ratioi);
          if (i==mx-1) li = nxi-1;
          lj = j - ratioj * (j/ratioj);
          if (j==my-1) lj = neta-1;
          lk = l - ratiok * (l/ratiok);
          if (l==mz-1) lk = nzeta-1;

          /* corners */
          col     = (m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c)+m_ghost_c*(j_c-j_start_ghost_c)+(i_c-i_start_ghost_c));
          cols[0] = idx_c[col];
          Ni[0]   = 1.0;
          if ((li==0) || (li==nxi-1)) {
            if ((lj==0) || (lj==neta-1)) {
              if ((lk==0) || (lk==nzeta-1)) {
                ierr = MatSetValue(mat,row,cols[0],Ni[0],INSERT_VALUES);CHKERRQ(ierr);
                continue;
              }
            }
          }

          /* edges + interior */
          /* remainders */
          if (i==mx-1) i_c--;
          if (j==my-1) j_c--;
          if (l==mz-1) l_c--;

          col     = (m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c));
          cols[0] = idx_c[col]; /* one left and below; or we are right on it */
          cols[1] = idx_c[col+1]; /* one right and below */
          cols[2] = idx_c[col+m_ghost_c];  /* one left and above */
          cols[3] = idx_c[col+(m_ghost_c+1)]; /* one right and above */

          cols[4] = idx_c[col+m_ghost_c*n_ghost_c]; /* one left and below and front; or we are right on it */
          cols[5] = idx_c[col+(m_ghost_c*n_ghost_c+1)]; /* one right and below, and front */
          cols[6] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c)]; /* one left and above and front*/
          cols[7] = idx_c[col+(m_ghost_c*n_ghost_c+m_ghost_c+1)]; /* one right and above and front */

          Ni[0] = 0.125*(1.0-xi[li])*(1.0-eta[lj])*(1.0-zeta[lk]);
          Ni[1] = 0.125*(1.0+xi[li])*(1.0-eta[lj])*(1.0-zeta[lk]);
          Ni[2] = 0.125*(1.0-xi[li])*(1.0+eta[lj])*(1.0-zeta[lk]);
          Ni[3] = 0.125*(1.0+xi[li])*(1.0+eta[lj])*(1.0-zeta[lk]);

          Ni[4] = 0.125*(1.0-xi[li])*(1.0-eta[lj])*(1.0+zeta[lk]);
          Ni[5] = 0.125*(1.0+xi[li])*(1.0-eta[lj])*(1.0+zeta[lk]);
          Ni[6] = 0.125*(1.0-xi[li])*(1.0+eta[lj])*(1.0+zeta[lk]);
          Ni[7] = 0.125*(1.0+xi[li])*(1.0+eta[lj])*(1.0+zeta[lk]);

          for (n=0; n<8; n++) {
            if (PetscAbsScalar(Ni[n])<1.0e-32) cols[n]=-1;
          }
          ierr = MatSetValues(mat,1,&row,8,cols,Ni,INSERT_VALUES);CHKERRQ(ierr);

        }
      }
    }
    ierr = PetscFree(xi);CHKERRQ(ierr);
    ierr = PetscFree(eta);CHKERRQ(ierr);
    ierr = PetscFree(zeta);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateMAIJ(mat,dof,A);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateInterpolation_DA(DM dac,DM daf,Mat *A,Vec *scale)
{
  PetscErrorCode   ierr;
  PetscInt         dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DMBoundaryType   bxc,byc,bzc,bxf,byf,bzf;
  DMDAStencilType  stc,stf;
  DM_DA            *ddc = (DM_DA*)dac->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DM_CLASSID,1);
  PetscValidHeaderSpecific(daf,DM_CLASSID,2);
  PetscValidPointer(A,3);
  if (scale) PetscValidPointer(scale,4);

  ierr = DMDAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&bxc,&byc,&bzc,&stc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&bxf,&byf,&bzf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Dimensions of DMDA do not match %D %D",dimc,dimf);
  if (dofc != doff) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"DOF of DMDA do not match %D %D",dofc,doff);
  if (sc != sf) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Stencil width of DMDA do not match %D %D",sc,sf);
  if (bxc != bxf || byc != byf || bzc != bzf) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Boundary type different in two DMDAs");
  if (stc != stf) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Stencil type different in two DMDAs");
  if (Mc < 2 && Mf > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2 && Nf > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2 && Pf > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in z direction");

  if (ddc->interptype == DMDA_Q1) {
    if (dimc == 1) {
      ierr = DMCreateInterpolation_DA_1D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 2) {
      ierr = DMCreateInterpolation_DA_2D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 3) {
      ierr = DMCreateInterpolation_DA_3D_Q1(dac,daf,A);CHKERRQ(ierr);
    } else SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_SUP,"No support for this DMDA dimension %D for interpolation type %d",dimc,(int)ddc->interptype);
  } else if (ddc->interptype == DMDA_Q0) {
    if (dimc == 1) {
      ierr = DMCreateInterpolation_DA_1D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 2) {
      ierr = DMCreateInterpolation_DA_2D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else if (dimc == 3) {
      ierr = DMCreateInterpolation_DA_3D_Q0(dac,daf,A);CHKERRQ(ierr);
    } else SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_SUP,"No support for this DMDA dimension %D for interpolation type %d",dimc,(int)ddc->interptype);
  }
  if (scale) {
    ierr = DMCreateInterpolationScale((DM)dac,(DM)daf,*A,scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInjection_DA_1D(DM dac,DM daf,VecScatter *inject)
{
  PetscErrorCode         ierr;
  PetscInt               i,i_start,m_f,Mx,dof;
  const PetscInt         *idx_f;
  ISLocalToGlobalMapping ltog_f;
  PetscInt               m_ghost,m_ghost_c;
  PetscInt               row,i_start_ghost,mx,m_c,nc,ratioi;
  PetscInt               i_start_c,i_start_ghost_c;
  PetscInt               *cols;
  DMBoundaryType         bx;
  Vec                    vecf,vecc;
  IS                     isf;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&bx,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (bx == DM_BOUNDARY_PERIODIC) {
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }

  ierr = DMDAGetCorners(daf,&i_start,NULL,NULL,&m_f,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,NULL,NULL,&m_ghost,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,NULL,NULL,&m_c,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,NULL,NULL,&m_ghost_c,NULL,NULL);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  nc   = 0;
  ierr = PetscMalloc1(m_f,&cols);CHKERRQ(ierr);

  for (i=i_start_c; i<i_start_c+m_c; i++) {
    PetscInt i_f = i*ratioi;

    if (i_f < i_start_ghost || i_f >= i_start_ghost+m_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\ni_c %D i_f %D fine ghost range [%D,%D]",i,i_f,i_start_ghost,i_start_ghost+m_ghost);

    row        = idx_f[(i_f-i_start_ghost)];
    cols[nc++] = row;
  }

  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISCreateBlock(PetscObjectComm((PetscObject)daf),dof,nc,cols,PETSC_OWN_POINTER,&isf);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = VecScatterCreate(vecf,isf,vecc,NULL,inject);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = ISDestroy(&isf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInjection_DA_2D(DM dac,DM daf,VecScatter *inject)
{
  PetscErrorCode         ierr;
  PetscInt               i,j,i_start,j_start,m_f,n_f,Mx,My,dof;
  const PetscInt         *idx_c,*idx_f;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  PetscInt               m_ghost,n_ghost,m_ghost_c,n_ghost_c;
  PetscInt               row,i_start_ghost,j_start_ghost,mx,m_c,my,nc,ratioi,ratioj;
  PetscInt               i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  PetscInt               *cols;
  DMBoundaryType         bx,by;
  Vec                    vecf,vecc;
  IS                     isf;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,&My,NULL,NULL,NULL,NULL,NULL,NULL,&bx,&by,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,&my,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (bx == DM_BOUNDARY_PERIODIC) {
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (by == DM_BOUNDARY_PERIODIC) {
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }

  ierr = DMDAGetCorners(daf,&i_start,&j_start,NULL,&m_f,&n_f,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,NULL,&m_ghost,&n_ghost,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,NULL,&m_c,&n_c,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,NULL,&m_ghost_c,&n_ghost_c,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  nc   = 0;
  ierr = PetscMalloc1(n_f*m_f,&cols);CHKERRQ(ierr);
  for (j=j_start_c; j<j_start_c+n_c; j++) {
    for (i=i_start_c; i<i_start_c+m_c; i++) {
      PetscInt i_f = i*ratioi,j_f = j*ratioj;
      if (j_f < j_start_ghost || j_f >= j_start_ghost+n_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    j_c %D j_f %D fine ghost range [%D,%D]",j,j_f,j_start_ghost,j_start_ghost+n_ghost);
      if (i_f < i_start_ghost || i_f >= i_start_ghost+m_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA\n\
    i_c %D i_f %D fine ghost range [%D,%D]",i,i_f,i_start_ghost,i_start_ghost+m_ghost);
      row        = idx_f[(m_ghost*(j_f-j_start_ghost) + (i_f-i_start_ghost))];
      cols[nc++] = row;
    }
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  ierr = ISCreateBlock(PetscObjectComm((PetscObject)daf),dof,nc,cols,PETSC_OWN_POINTER,&isf);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = VecScatterCreate(vecf,isf,vecc,NULL,inject);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = ISDestroy(&isf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateInjection_DA_3D(DM dac,DM daf,VecScatter *inject)
{
  PetscErrorCode         ierr;
  PetscInt               i,j,k,i_start,j_start,k_start,m_f,n_f,p_f,Mx,My,Mz;
  PetscInt               m_ghost,n_ghost,p_ghost,m_ghost_c,n_ghost_c,p_ghost_c;
  PetscInt               i_start_ghost,j_start_ghost,k_start_ghost;
  PetscInt               mx,my,mz,ratioi,ratioj,ratiok;
  PetscInt               i_start_c,j_start_c,k_start_c;
  PetscInt               m_c,n_c,p_c;
  PetscInt               i_start_ghost_c,j_start_ghost_c,k_start_ghost_c;
  PetscInt               row,nc,dof;
  const PetscInt         *idx_c,*idx_f;
  ISLocalToGlobalMapping ltog_f,ltog_c;
  PetscInt               *cols;
  DMBoundaryType         bx,by,bz;
  Vec                    vecf,vecc;
  IS                     isf;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(dac,NULL,&Mx,&My,&Mz,NULL,NULL,NULL,NULL,NULL,&bx,&by,&bz,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,NULL,&mx,&my,&mz,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  if (bx == DM_BOUNDARY_PERIODIC) {
    ratioi = mx/Mx;
    if (ratioi*Mx != mx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mx/Mx  must be integer: mx %D Mx %D",mx,Mx);
  } else {
    ratioi = (mx-1)/(Mx-1);
    if (ratioi*(Mx-1) != mx-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mx - 1)/(Mx - 1) must be integer: mx %D Mx %D",mx,Mx);
  }
  if (by == DM_BOUNDARY_PERIODIC) {
    ratioj = my/My;
    if (ratioj*My != my) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: my/My  must be integer: my %D My %D",my,My);
  } else {
    ratioj = (my-1)/(My-1);
    if (ratioj*(My-1) != my-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (my - 1)/(My - 1) must be integer: my %D My %D",my,My);
  }
  if (bz == DM_BOUNDARY_PERIODIC) {
    ratiok = mz/Mz;
    if (ratiok*Mz != mz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: mz/Mz  must be integer: mz %D My %D",mz,Mz);
  } else {
    ratiok = (mz-1)/(Mz-1);
    if (ratiok*(Mz-1) != mz-1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Ratio between levels: (mz - 1)/(Mz - 1) must be integer: mz %D Mz %D",mz,Mz);
  }

  ierr = DMDAGetCorners(daf,&i_start,&j_start,&k_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&k_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(daf,&ltog_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&k_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&k_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dac,&ltog_c);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  nc   = 0;
  ierr = PetscMalloc1(n_f*m_f*p_f,&cols);CHKERRQ(ierr);
  for (k=k_start_c; k<k_start_c+p_c; k++) {
    for (j=j_start_c; j<j_start_c+n_c; j++) {
      for (i=i_start_c; i<i_start_c+m_c; i++) {
        PetscInt i_f = i*ratioi,j_f = j*ratioj,k_f = k*ratiok;
        if (k_f < k_start_ghost || k_f >= k_start_ghost+p_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA  "
                                                                          "k_c %D k_f %D fine ghost range [%D,%D]",k,k_f,k_start_ghost,k_start_ghost+p_ghost);
        if (j_f < j_start_ghost || j_f >= j_start_ghost+n_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA  "
                                                                          "j_c %D j_f %D fine ghost range [%D,%D]",j,j_f,j_start_ghost,j_start_ghost+n_ghost);
        if (i_f < i_start_ghost || i_f >= i_start_ghost+m_ghost) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Processor's coarse DMDA must lie over fine DMDA  "
                                                                          "i_c %D i_f %D fine ghost range [%D,%D]",i,i_f,i_start_ghost,i_start_ghost+m_ghost);
        row        = idx_f[(m_ghost*n_ghost*(k_f-k_start_ghost) + m_ghost*(j_f-j_start_ghost) + (i_f-i_start_ghost))];
        cols[nc++] = row;
      }
    }
  }
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_f,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreBlockIndices(ltog_c,&idx_c);CHKERRQ(ierr);

  ierr = ISCreateBlock(PetscObjectComm((PetscObject)daf),dof,nc,cols,PETSC_OWN_POINTER,&isf);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = VecScatterCreate(vecf,isf,vecc,NULL,inject);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dac,&vecc);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daf,&vecf);CHKERRQ(ierr);
  ierr = ISDestroy(&isf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  DMCreateInjection_DA(DM dac,DM daf,Mat *mat)
{
  PetscErrorCode  ierr;
  PetscInt        dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc,dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DMBoundaryType  bxc,byc,bzc,bxf,byf,bzf;
  DMDAStencilType stc,stf;
  VecScatter      inject = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dac,DM_CLASSID,1);
  PetscValidHeaderSpecific(daf,DM_CLASSID,2);
  PetscValidPointer(mat,3);

  ierr = DMDAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&bxc,&byc,&bzc,&stc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&bxf,&byf,&bzf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Dimensions of DMDA do not match %D %D",dimc,dimf);
  if (dofc != doff) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"DOF of DMDA do not match %D %D",dofc,doff);
  if (sc != sf) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Stencil width of DMDA do not match %D %D",sc,sf);
  if (bxc != bxf || byc != byf || bzc != bzf) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Boundary type different in two DMDAs");
  if (stc != stf) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Stencil type different in two DMDAs");
  if (Mc < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in x direction");
  if (dimc > 1 && Nc < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in y direction");
  if (dimc > 2 && Pc < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Coarse grid requires at least 2 points in z direction");

  if (dimc == 1) {
    ierr = DMCreateInjection_DA_1D(dac,daf,&inject);CHKERRQ(ierr);
  } else if (dimc == 2) {
    ierr = DMCreateInjection_DA_2D(dac,daf,&inject);CHKERRQ(ierr);
  } else if (dimc == 3) {
    ierr = DMCreateInjection_DA_3D(dac,daf,&inject);CHKERRQ(ierr);
  }
  ierr = MatCreateScatter(PetscObjectComm((PetscObject)inject), inject, mat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&inject);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   DMCreateAggregates - Deprecated, see DMDACreateAggregates.

   Level: intermediate
@*/
PetscErrorCode DMCreateAggregates(DM dac,DM daf,Mat *mat)
{
  return DMDACreateAggregates(dac,daf,mat);
}

/*@
   DMDACreateAggregates - Gets the aggregates that map between
   grids associated with two DMDAs.

   Collective on dmc

   Input Parameters:
+  dmc - the coarse grid DMDA
-  dmf - the fine grid DMDA

   Output Parameters:
.  rest - the restriction matrix (transpose of the projection matrix)

   Level: intermediate

   Note: This routine is not used by PETSc.
   It is not clear what its use case is and it may be removed in a future release.
   Users should contact petsc-maint@mcs.anl.gov if they plan to use it.

.seealso: DMRefine(), DMCreateInjection(), DMCreateInterpolation()
@*/
PetscErrorCode DMDACreateAggregates(DM dac,DM daf,Mat *rest)
{
  PetscErrorCode         ierr;
  PetscInt               dimc,Mc,Nc,Pc,mc,nc,pc,dofc,sc;
  PetscInt               dimf,Mf,Nf,Pf,mf,nf,pf,doff,sf;
  DMBoundaryType         bxc,byc,bzc,bxf,byf,bzf;
  DMDAStencilType        stc,stf;
  PetscInt               i,j,l;
  PetscInt               i_start,j_start,l_start, m_f,n_f,p_f;
  PetscInt               i_start_ghost,j_start_ghost,l_start_ghost,m_ghost,n_ghost,p_ghost;
  const PetscInt         *idx_f;
  PetscInt               i_c,j_c,l_c;
  PetscInt               i_start_c,j_start_c,l_start_c, m_c,n_c,p_c;
  PetscInt               i_start_ghost_c,j_start_ghost_c,l_start_ghost_c,m_ghost_c,n_ghost_c,p_ghost_c;
  const PetscInt         *idx_c;
  PetscInt               d;
  PetscInt               a;
  PetscInt               max_agg_size;
  PetscInt               *fine_nodes;
  PetscScalar            *one_vec;
  PetscInt               fn_idx;
  ISLocalToGlobalMapping ltogmf,ltogmc;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dac,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecificType(daf,DM_CLASSID,2,DMDA);
  PetscValidPointer(rest,3);

  ierr = DMDAGetInfo(dac,&dimc,&Mc,&Nc,&Pc,&mc,&nc,&pc,&dofc,&sc,&bxc,&byc,&bzc,&stc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(daf,&dimf,&Mf,&Nf,&Pf,&mf,&nf,&pf,&doff,&sf,&bxf,&byf,&bzf,&stf);CHKERRQ(ierr);
  if (dimc != dimf) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Dimensions of DMDA do not match %D %D",dimc,dimf);
  if (dofc != doff) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"DOF of DMDA do not match %D %D",dofc,doff);
  if (sc != sf) SETERRQ2(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Stencil width of DMDA do not match %D %D",sc,sf);
  if (bxc != bxf || byc != byf || bzc != bzf) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Boundary type different in two DMDAs");
  if (stc != stf) SETERRQ(PetscObjectComm((PetscObject)daf),PETSC_ERR_ARG_INCOMP,"Stencil type different in two DMDAs");

  if (Mf < Mc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Coarse grid has more points than fine grid, Mc %D, Mf %D", Mc, Mf);
  if (Nf < Nc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Coarse grid has more points than fine grid, Nc %D, Nf %D", Nc, Nf);
  if (Pf < Pc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Coarse grid has more points than fine grid, Pc %D, Pf %D", Pc, Pf);

  if (Pc < 0) Pc = 1;
  if (Pf < 0) Pf = 1;
  if (Nc < 0) Nc = 1;
  if (Nf < 0) Nf = 1;

  ierr = DMDAGetCorners(daf,&i_start,&j_start,&l_start,&m_f,&n_f,&p_f);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(daf,&i_start_ghost,&j_start_ghost,&l_start_ghost,&m_ghost,&n_ghost,&p_ghost);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(daf,&ltogmf);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogmf,&idx_f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dac,&i_start_c,&j_start_c,&l_start_c,&m_c,&n_c,&p_c);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dac,&i_start_ghost_c,&j_start_ghost_c,&l_start_ghost_c,&m_ghost_c,&n_ghost_c,&p_ghost_c);CHKERRQ(ierr);

  ierr = DMGetLocalToGlobalMapping(dac,&ltogmc);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(ltogmc,&idx_c);CHKERRQ(ierr);

  /*
     Basic idea is as follows. Here's a 2D example, suppose r_x, r_y are the ratios
     for dimension 1 and 2 respectively.
     Let (i,j) be a coarse grid node. All the fine grid nodes between r_x*i and r_x*(i+1)
     and r_y*j and r_y*(j+1) will be grouped into the same coarse grid agregate.
     Each specific dof on the fine grid is mapped to one dof on the coarse grid.
  */

  max_agg_size = (Mf/Mc+1)*(Nf/Nc+1)*(Pf/Pc+1);

  /* create the matrix that will contain the restriction operator */
  ierr = MatCreateAIJ(PetscObjectComm((PetscObject)daf), m_c*n_c*p_c*dofc, m_f*n_f*p_f*doff, Mc*Nc*Pc*dofc, Mf*Nf*Pf*doff,
                      max_agg_size, NULL, max_agg_size, NULL, rest);CHKERRQ(ierr);

  /* store nodes in the fine grid here */
  ierr = PetscMalloc2(max_agg_size, &one_vec,max_agg_size, &fine_nodes);CHKERRQ(ierr);
  for (i=0; i<max_agg_size; i++) one_vec[i] = 1.0;

  /* loop over all coarse nodes */
  for (l_c=l_start_c; l_c<l_start_c+p_c; l_c++) {
    for (j_c=j_start_c; j_c<j_start_c+n_c; j_c++) {
      for (i_c=i_start_c; i_c<i_start_c+m_c; i_c++) {
        for (d=0; d<dofc; d++) {
          /* convert to local "natural" numbering and then to PETSc global numbering */
          a = idx_c[dofc*(m_ghost_c*n_ghost_c*(l_c-l_start_ghost_c) + m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c))] + d;

          fn_idx = 0;
          /* Corresponding fine points are all points (i_f, j_f, l_f) such that
             i_c*Mf/Mc <= i_f < (i_c+1)*Mf/Mc
             (same for other dimensions)
          */
          for (l=l_c*Pf/Pc; l<PetscMin((l_c+1)*Pf/Pc,Pf); l++) {
            for (j=j_c*Nf/Nc; j<PetscMin((j_c+1)*Nf/Nc,Nf); j++) {
              for (i=i_c*Mf/Mc; i<PetscMin((i_c+1)*Mf/Mc,Mf); i++) {
                fine_nodes[fn_idx] = idx_f[doff*(m_ghost*n_ghost*(l-l_start_ghost) + m_ghost*(j-j_start_ghost) + (i-i_start_ghost))] + d;
                fn_idx++;
              }
            }
          }
          /* add all these points to one aggregate */
          ierr = MatSetValues(*rest, 1, &a, fn_idx, fine_nodes, one_vec, INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(ltogmf,&idx_f);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRestoreIndices(ltogmc,&idx_c);CHKERRQ(ierr);
  ierr = PetscFree2(one_vec,fine_nodes);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
