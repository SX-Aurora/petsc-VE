
#ifdef __ve__
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
//#define VE_SETERRQ
#ifndef BUF_SIZE
#define BUF_SIZE 8192*4
#endif
#ifndef MAX_CNT
#define MAX_CNT 5
#endif
#endif

/*   DMDA/KSP solving a system of linear equations.
     Poisson equation in 2D:

     div(grad p) = f,  0 < x,y < 1
     with
       forcing function f = -cos(m*pi*x)*cos(n*pi*y),
       Neuman boundary conditions
        dp/dx = 0 for x = 0, x = 1.
        dp/dy = 0 for y = 0, y = 1.

     Contributed by Michael Boghosian <boghmic@iit.edu>, 2008,
         based on petsc/src/ksp/ksp/tutorials/ex29.c and ex32.c

     Compare to ex66.c

     Example of Usage:
          ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 3 -ksp_monitor -ksp_view -dm_view draw -draw_pause -1
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg  -pc_mg_levels 1 -mg_levels_0_pc_type ilu -mg_levels_0_pc_factor_levels 1 -ksp_monitor -ksp_view
          ./ex50 -da_grid_x 100 -da_grid_y 100 -pc_type mg -pc_mg_levels 1 -mg_levels_0_pc_type lu -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor
          mpiexec -n 4 ./ex50 -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 10 -ksp_monitor -ksp_view -log_view
*/

static char help[] = "Solves 2D Poisson equation using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>

extern PetscErrorCode ComputeJacobian(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

typedef struct {
  PetscScalar uu, tt;
} UserContext;

int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  UserContext    user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,11,11,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,(DM)da);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);

  user.uu     = 1.0;
  user.tt     = 1.0;

  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeJacobian,&user);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,M,N,xm,ym,xs,ys;
  PetscScalar    Hx,Hy,pi,uu,tt;
  PetscScalar    **array;
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  uu   = user->uu; tt = user->tt;
  pi   = 4*atan(1.0);
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);

  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr); /* Fine grid */
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = -PetscCosScalar(uu*pi*((PetscReal)i+0.5)*Hx)*PetscCosScalar(tt*pi*((PetscReal)j+0.5)*Hy)*Hx*Hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef __ve__
#define MatSetValues_SeqAIJ_A_Private1(row,col,value,addv,orow,ocol)     \
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
          goto a_noinsert1; \
        } \
      }  \
      if (value == 0.0 && ignorezeroentries && row != col) {low1 = 0; high1 = nrow1;goto a_noinsert1;} \
      if (nonew == 1) {low1 = 0; high1 = nrow1; goto a_noinsert1;}                \
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
      MatSeqXAIJReallocateAIJ(A,am,1,nrow1,row,col,rmax1,aa,ai,aj,rp1,ap1,aimax,nonew,MatScalar); \
      N = nrow1++ - 1; a->nz++; high1++; \
      /* shift up all the later entries in this row */ \
      ierr = PetscArraymove(rp1+_i+1,rp1+_i,N-_i+1);CHKERRQ(ierr);\
      ierr = PetscArraymove(ap1+_i+1,ap1+_i,N-_i+1);CHKERRQ(ierr);\
      rp1[_i] = col;  \
      ap1[_i] = value;  \
      A->nonzerostate++;\
      a_noinsert1: ; \
      ailen[row] = nrow1; \
}

#define MatSetValues_SeqAIJ_B_Private1(row,col,value,addv,orow,ocol) \
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
        goto b_noinsert1;                                  \
      }                                                   \
    }                                                     \
    if (value == 0.0 && ignorezeroentries) {low2 = 0; high2 = nrow2; goto b_noinsert1;} \
    if (nonew == 1) {low2 = 0; high2 = nrow2; goto b_noinsert1;}                        \
    if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B,bm,1,nrow2,row,col,rmax2,ba,bi,bj,rp2,ap2,bimax,nonew,MatScalar); \
    N = nrow2++ - 1; b->nz++; high2++;                    \
    /* shift up all the later entries in this row */      \
    ierr = PetscArraymove(rp2+_i+1,rp2+_i,N-_i+1);CHKERRQ(ierr);\
    ierr = PetscArraymove(ap2+_i+1,ap2+_i,N-_i+1);CHKERRQ(ierr);\
    rp2[_i] = col;                                        \
    ap2[_i] = value;                                      \
    B->nonzerostate++;                                    \
    b_noinsert1: ;                                         \
    bilen[row] = nrow2;                                   \
  }
#define MatSetValues_SeqAIJ_A_Private2(row,col,value,addv,orow,ocol)     \
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
          goto a_noinsert2; \
        } \
      }  \
      if (value == 0.0 && ignorezeroentries && row != col) {low1 = 0; high1 = nrow1;goto a_noinsert2;} \
      if (nonew == 1) {low1 = 0; high1 = nrow1; goto a_noinsert2;}                \
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
      MatSeqXAIJReallocateAIJ(A,am,1,nrow1,row,col,rmax1,aa,ai,aj,rp1,ap1,aimax,nonew,MatScalar); \
      N = nrow1++ - 1; a->nz++; high1++; \
      /* shift up all the later entries in this row */ \
      ierr = PetscArraymove(rp1+_i+1,rp1+_i,N-_i+1);CHKERRQ(ierr);\
      ierr = PetscArraymove(ap1+_i+1,ap1+_i,N-_i+1);CHKERRQ(ierr);\
      rp1[_i] = col;  \
      ap1[_i] = value;  \
      A->nonzerostate++;\
      a_noinsert2: ; \
      ailen[row] = nrow1; \
}

#define MatSetValues_SeqAIJ_B_Private2(row,col,value,addv,orow,ocol) \
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
        goto b_noinsert2;                                  \
      }                                                   \
    }                                                     \
    if (value == 0.0 && ignorezeroentries) {low2 = 0; high2 = nrow2; goto b_noinsert2;} \
    if (nonew == 1) {low2 = 0; high2 = nrow2; goto b_noinsert2;}                        \
    if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B,bm,1,nrow2,row,col,rmax2,ba,bi,bj,rp2,ap2,bimax,nonew,MatScalar); \
    N = nrow2++ - 1; b->nz++; high2++;                    \
    /* shift up all the later entries in this row */      \
    ierr = PetscArraymove(rp2+_i+1,rp2+_i,N-_i+1);CHKERRQ(ierr);\
    ierr = PetscArraymove(ap2+_i+1,ap2+_i,N-_i+1);CHKERRQ(ierr);\
    rp2[_i] = col;                                        \
    ap2[_i] = value;                                      \
    B->nonzerostate++;                                    \
    b_noinsert2: ;                                         \
    bilen[row] = nrow2;                                   \
  }
#endif

PetscErrorCode ComputeJacobian(KSP ksp,Mat J, Mat jac,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm, ym, xs, ys, num, numi, numj;
  PetscScalar    v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr  = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr  = DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  ierr  = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
#ifdef __ve__
 PetscInt buf_tb1[BUF_SIZE];
 PetscInt buf_tb2[BUF_SIZE];
 PetscInt num_tb[xm], numi_tb[xm], numj_tb[xm], loop_flg[xm], jj;
 MatStencil col_tb[5*xm], row_tb[xm];
 PetscScalar v_tb[5*xm];
 PetscBool flg,flg2;
 ierr = MatSetValues_IsMPIAIJ(jac,&flg);
 ierr = MatSetValues_IsSeqAIJ(jac,&flg2);
 if (jac->assembled) {
   jac->was_assembled = PETSC_TRUE;
   jac->assembled     = PETSC_FALSE;
 }
 if (jac->insertmode == NOT_SET_VALUES) {
   jac->insertmode = INSERT_VALUES;
 } else if (PetscUnlikely(jac->insertmode != INSERT_VALUES)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
 if (flg==PETSC_TRUE && (1+5)*xm <= BUF_SIZE && !jac->ops->setvalueslocal && !PetscDefined(USE_DEBUG)) {
  if (jac->was_assembled) {
    if (!((Mat_MPIAIJ*)jac->data)->colmap) {
      ierr = MatCreateColmap_MPIAIJ_Private(jac);CHKERRQ(ierr);
    }
  }
  for (j=ys; j<ys+ym; j++) {
   for (jj=0; jj<MAX_CNT; jj++) {
    for (i=xs; i<xs+xm; i++) {
      loop_flg[i-xs] = 0;
    }
#pragma _NEC ivdep
    for (i=xs; i<xs+xm; i++) {
     if (jj==0) {
      row_tb[i-xs].i = i; row_tb[i-xs].j = j;
      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num_tb[i-xs]=0; numi_tb[i-xs]=0; numj_tb[i-xs]=0;
        if (j!=0) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HxdHy; col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j-1;
          num_tb[i-xs]++; numj_tb[i-xs]++;
        }
        if (i!=0) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HydHx; col_tb[num_tb[i-xs]+5*(i-xs)].i = i-1; col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
          num_tb[i-xs]++; numi_tb[i-xs]++;
        }
        if (i!=M-1) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HydHx; col_tb[num_tb[i-xs]+5*(i-xs)].i = i+1; col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
          num_tb[i-xs]++; numi_tb[i-xs]++;
        }
        if (j!=N-1) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HxdHy; col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j+1;
          num_tb[i-xs]++; numj_tb[i-xs]++;
        }
        v_tb[num_tb[i-xs]*xm+i-xs] = ((PetscReal)(numj_tb[i-xs])*HxdHy + (PetscReal)(numi_tb[i-xs])*HydHx); col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
        num_tb[i-xs]++;
        //ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        v_tb[0*xm+i-xs] = -HxdHy;              col_tb[0+5*(i-xs)].i = i;   col_tb[0+5*(i-xs)].j = j-1;
        v_tb[1*xm+i-xs] = -HydHx;              col_tb[1+5*(i-xs)].i = i-1; col_tb[1+5*(i-xs)].j = j;
        v_tb[2*xm+i-xs] = 2.0*(HxdHy + HydHx); col_tb[2+5*(i-xs)].i = i;   col_tb[2+5*(i-xs)].j = j;
        v_tb[3*xm+i-xs] = -HydHx;              col_tb[3+5*(i-xs)].i = i+1; col_tb[3+5*(i-xs)].j = j;
        v_tb[4*xm+i-xs] = -HxdHy;              col_tb[4+5*(i-xs)].i = i;   col_tb[4+5*(i-xs)].j = j+1;
        //ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
     }
        //PetscErrorCode MatSetValuesStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],const PetscScalar v[],InsertMode addv)
        //-- enter --
        {
          PetscErrorCode ierr;
          PetscInt m;
          Mat mat;
          const MatStencil *idxm;
          const MatStencil *idxn;
          mat = jac;
          idxm = &row_tb[i-xs];
          idxn = &col_tb[5*(i-xs)];
          PetscInt       *bufm=NULL,*bufn=NULL,*jdxm,*jdxn;
          PetscInt       j_,i_,dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
          PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

          PetscInt n;
          InsertMode addv;
        
          m = 1;
          if (i==0 || j==0 || i==M-1 || j==N-1) {
            n = num_tb[i-xs];
          } else {
            n = 5;
          }
          addv = INSERT_VALUES;
 
          PetscFunctionBegin;
         if (m && n) { /* else : no values to insert */
          PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
          PetscValidType(mat,1);
          PetscValidPointer(idxm,3);
          PetscValidPointer(idxn,5);
        
          jdxm = buf_tb1; jdxn = buf_tb1+m*xm;

         if (jj==0) {
          i_=0;
          if (i_<m) { // max(m)=1
            j_=0;
            if (j_<3-sdim) { dxm++; j_++; } // max(3-sdim)=1
            tmp = *dxm++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxm++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxm-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxm++;
            jdxm[i_*xm+i-xs] = tmp;
            i_++;
          }
          i_=0;
          if (i_<n) { // max(n)=1
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=2
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=3
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=4
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=5
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
         }
          //ierr = MatSetValuesLocal(mat,m,jdxm,n,jdxn,v,addv);CHKERRQ(ierr);
          //PetscErrorCode MatSetValuesLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
          //-- enter --
          {
            PetscErrorCode ierr;
            PetscInt nrow;
            const PetscInt *irow;
            PetscInt ncol;
            const PetscInt *icol;
            const PetscScalar *y;
          
            nrow = m;
            irow = jdxm;
            ncol = n;
            icol = jdxn;
            y = v;
          
            PetscFunctionBeginHot;
            PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
            PetscValidType(mat,1);
            MatCheckPreallocated(mat,1);
           if (nrow && ncol) { /* else : no values to insert */
            PetscValidIntPointer(irow,3);
            PetscValidIntPointer(icol,5);
#ifdef VE_SETERRQ
            else if (PetscUnlikely(mat->insertmode != addv)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
            if (PetscDefined(USE_DEBUG)) {
              if (mat->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
              if (!mat->ops->setvalueslocal && !mat->ops->setvalues) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
            }
#endif
              PetscInt *bufr=NULL,*bufc=NULL,*irowm,*icolm;
              irowm = buf_tb2; icolm = buf_tb2+nrow*xm;
#ifdef VE_SETERRQ
              if (!mat->rmap->mapping) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatSetValuesLocal() cannot proceed without local-to-global row mapping (See MatSetLocalToGlobalMapping()).");
              if (!mat->cmap->mapping) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatSetValuesLocal() cannot proceed without local-to-global column mapping (See MatSetLocalToGlobalMapping()).");
#endif
             if (jj==0) {
              //ierr = ISLocalToGlobalMappingApply(mat->rmap->mapping,nrow,irow,irowm);CHKERRQ(ierr);
              //PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
              //-- enter --
              {
                PetscInt i_,bs,Nmax;
                ISLocalToGlobalMapping mapping;
                PetscInt N;
                const PetscInt *in;
                PetscInt *out;
          
                mapping = mat->rmap->mapping;
                N = nrow;
                in = irow;
                out = irowm;
          
                PetscFunctionBegin;
                PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
                bs   = mapping->bs;
                Nmax = bs*mapping->n;
                if (bs == 1) {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                } else {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                }
                //PetscFunctionReturn(0);
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
              //ierr = ISLocalToGlobalMappingApply(mat->cmap->mapping,ncol,icol,icolm);CHKERRQ(ierr);
              //PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
              //-- enter --
              {
                PetscInt i_,bs,Nmax;
                ISLocalToGlobalMapping mapping;
                PetscInt N;
                const PetscInt *in;
                PetscInt *out;
          
                mapping = mat->cmap->mapping;
                N = ncol;
                in = icol;
                out = icolm;
          
                PetscFunctionBegin;
                PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
                bs   = mapping->bs;
                Nmax = bs*mapping->n;
                if (bs == 1) {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=2
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*(xs+xm)+i);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=3
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=4
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=5
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                } else {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=2
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=3
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=4
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=5
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                }
                //PetscFunctionReturn(0);
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
             }
              //ierr = MatSetValues(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
              //PetscErrorCode MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
              //-- enter --
              {
                PetscErrorCode ierr;
                PetscInt m;
                const PetscInt *idxm;
                PetscInt n;
                const PetscInt *idxn;
                const PetscScalar *v;
              
                m = nrow;
                idxm = irowm;
                n = ncol;
                idxn = icolm;
                v = y;
              
                PetscFunctionBeginHot;
                PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
                PetscValidType(mat,1);
               if (m && n) { /* else : no values to insert */
                PetscValidIntPointer(idxm,3);
                PetscValidIntPointer(idxn,5);
                MatCheckPreallocated(mat,1);

                //ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
                //ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
                //PetscErrorCode MatSetValues_MPIAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
                //-- enter --
                {
                  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
                  PetscScalar    value = 0.0;
                  PetscErrorCode ierr;
                  PetscInt       i_,rstart  = mat->rmap->rstart,rend = mat->rmap->rend;
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
                  i_=0;
                  if (i_<m) { // max(m)=max(nrow)=1
                    if (im[i_*xm+i-xs] < 0) /* continue */ ; else
                    {
#ifdef VE_SETERRQ
                    if (PetscUnlikely(im[i_*xm+i-xs] >= mat->rmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i_*xm+i-xs],mat->rmap->N-1);
#endif
                    if (im[i_*xm+i-xs] >= rstart && im[i_*xm+i-xs] < rend) {
                      row      = im[i_*xm+i-xs] - rstart;
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
                
                      if (jj<n) { // max(n)=max(ncol)=number of jj loop
                        if (v_tb)  value = roworiented ? v_tb[(i_*n+jj)*xm+i-xs] : v_tb[(i_+jj*m)*xm+i-xs];
                        if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES) && im[i_*xm+i-xs] != in[jj*xm+i-xs]) {
                          ; // continue;
                        } else
                        if (in[jj*xm+i-xs] >= cstart && in[jj*xm+i-xs] < cend) {
                          col   = in[jj*xm+i-xs] - cstart;
                          nonew = a->nonew;
                          //MatSetValues_SeqAIJ_A_Private1(row,col,value,addv,im[i_*xm+i-xs],in[jj*xm+i-xs]);
                          //#define MatSetValues_SeqAIJ_A_Private(row,col,value,addv,orow,ocol)
                          { //-- enter
                            PetscInt  orow,ocol;
                            PetscBool bflg, gflg; // bflg means break flag, gflg means goto flag
                            orow = im[i_*xm+i-xs];
                            ocol = in[jj*xm+i-xs];
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
                            if (_i<high1 && bflg==PETSC_FALSE) { // max(high1-low1+1)=5
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
                              // if nrow1<rmax1 do nothing
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
                        } else if (in[jj*xm+i-xs] < 0) {;} // continue;
                        else {
                          loop_flg[i-xs] = 1;
                        }
                      }
                    } else {
#ifdef VE_SETERRQ
                      if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[jj*xm+i-xs]);
#endif
                      //if (!aij->donotstash) {
                      //  mat->assembled = PETSC_FALSE;
                      //  if (roworiented) {
                      //    ierr = MatStashValuesRow_Private(&mat->stash,im[i_],n,in,v+i*n,(PetscBool)(ignorezeroentries && (addv == ADD_VALUES)));CHKERRQ(ierr);
                      //  } else {
                      //    ierr = MatStashValuesCol_Private(&mat->stash,im[i_],n,in,v+i,m,(PetscBool)(ignorezeroentries && (addv == ADD_VALUES)));CHKERRQ(ierr);
                      //  }
                      //}
                    }
                    }
                    i_++;
                  }
                  //PetscFunctionReturn(0);
                }
                //-- exit --
                ierr = 0;CHKERRQ(ierr);
                //ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
                //PetscFunctionReturn(0);
               }
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
            //ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
            //PetscFunctionReturn(0);
           }
          }
          //-- exit
          ierr = 0;CHKERRQ(ierr);
          //PetscFunctionReturn(0);
         }
        }
        //-- exit --
        ierr = 0;CHKERRQ(ierr);
    }
    for (i=xs; i<xs+xm; i++) {
      if (!loop_flg[i-xs]) continue;

     if (jj==0) {
      row_tb[i-xs].i = i; row_tb[i-xs].j = j;
      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num_tb[i-xs]=0; numi_tb[i-xs]=0; numj_tb[i-xs]=0;
        if (j!=0) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HxdHy; col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j-1;
          num_tb[i-xs]++; numj_tb[i-xs]++;
        }
        if (i!=0) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HydHx; col_tb[num_tb[i-xs]+5*(i-xs)].i = i-1; col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
          num_tb[i-xs]++; numi_tb[i-xs]++;
        }
        if (i!=M-1) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HydHx; col_tb[num_tb[i-xs]+5*(i-xs)].i = i+1; col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
          num_tb[i-xs]++; numi_tb[i-xs]++;
        }
        if (j!=N-1) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HxdHy; col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j+1;
          num_tb[i-xs]++; numj_tb[i-xs]++;
        }
        v_tb[num_tb[i-xs]*xm+i-xs] = ((PetscReal)(numj_tb[i-xs])*HxdHy + (PetscReal)(numi_tb[i-xs])*HydHx); col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
        num_tb[i-xs]++;
        //ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        v_tb[0*xm+i-xs] = -HxdHy;              col_tb[0+5*(i-xs)].i = i;   col_tb[0+5*(i-xs)].j = j-1;
        v_tb[1*xm+i-xs] = -HydHx;              col_tb[1+5*(i-xs)].i = i-1; col_tb[1+5*(i-xs)].j = j;
        v_tb[2*xm+i-xs] = 2.0*(HxdHy + HydHx); col_tb[2+5*(i-xs)].i = i;   col_tb[2+5*(i-xs)].j = j;
        v_tb[3*xm+i-xs] = -HydHx;              col_tb[3+5*(i-xs)].i = i+1; col_tb[3+5*(i-xs)].j = j;
        v_tb[4*xm+i-xs] = -HxdHy;              col_tb[4+5*(i-xs)].i = i;   col_tb[4+5*(i-xs)].j = j+1;
        //ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
     }
        //PetscErrorCode MatSetValuesStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],const PetscScalar v[],InsertMode addv)
        //-- enter --
        {
          PetscErrorCode ierr;
          PetscInt m;
          Mat mat;
          const MatStencil *idxm;
          const MatStencil *idxn;
          mat = jac;
          idxm = &row_tb[i-xs];
          idxn = &col_tb[5*(i-xs)];
          PetscInt       *bufm=NULL,*bufn=NULL,*jdxm,*jdxn;
          PetscInt       j_,i_,dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
          PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

          PetscInt n;
          InsertMode addv;
        
          m = 1;
          if (i==0 || j==0 || i==M-1 || j==N-1) {
            n = num_tb[i-xs];
          } else {
            n = 5;
          }
          addv = INSERT_VALUES;
 
          PetscFunctionBegin;
         if (m && n) { /* else : no values to insert */
          PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
          PetscValidType(mat,1);
          PetscValidPointer(idxm,3);
          PetscValidPointer(idxn,5);
        
          jdxm = buf_tb1; jdxn = buf_tb1+m*xm;

         if (jj==0) {
          i_=0;
          if (i_<m) { // max(m)=1
            j_=0;
            if (j_<3-sdim) { dxm++; j_++; } // max(3-sdim)=1
            tmp = *dxm++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxm++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxm-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxm++;
            jdxm[i_*xm+i-xs] = tmp;
            i_++;
          }
          i_=0;
          if (i_<n) { // max(n)=1
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=2
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=3
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=4
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=5
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
         }
          //ierr = MatSetValuesLocal(mat,m,jdxm,n,jdxn,v,addv);CHKERRQ(ierr);
          //PetscErrorCode MatSetValuesLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
          //-- enter --
          {
            PetscErrorCode ierr;
            PetscInt nrow;
            const PetscInt *irow;
            PetscInt ncol;
            const PetscInt *icol;
            const PetscScalar *y;
          
            nrow = m;
            irow = jdxm;
            ncol = n;
            icol = jdxn;
            y = v;
          
            PetscFunctionBeginHot;
            PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
            PetscValidType(mat,1);
            MatCheckPreallocated(mat,1);
           if (nrow && ncol) { /* else : no values to insert */
            PetscValidIntPointer(irow,3);
            PetscValidIntPointer(icol,5);
#ifdef VE_SETERRQ
            else if (PetscUnlikely(mat->insertmode != addv)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
            if (PetscDefined(USE_DEBUG)) {
              if (mat->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
              if (!mat->ops->setvalueslocal && !mat->ops->setvalues) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
            }
#endif
              PetscInt *bufr=NULL,*bufc=NULL,*irowm,*icolm;
              irowm = buf_tb2; icolm = buf_tb2+nrow*xm;
#ifdef VE_SETERRQ
              if (!mat->rmap->mapping) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatSetValuesLocal() cannot proceed without local-to-global row mapping (See MatSetLocalToGlobalMapping()).");
              if (!mat->cmap->mapping) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatSetValuesLocal() cannot proceed without local-to-global column mapping (See MatSetLocalToGlobalMapping()).");
#endif
             if (jj==0) {
              //ierr = ISLocalToGlobalMappingApply(mat->rmap->mapping,nrow,irow,irowm);CHKERRQ(ierr);
              //PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
              //-- enter --
              {
                PetscInt i_,bs,Nmax;
                ISLocalToGlobalMapping mapping;
                PetscInt N;
                const PetscInt *in;
                PetscInt *out;
          
                mapping = mat->rmap->mapping;
                N = nrow;
                in = irow;
                out = irowm;
          
                PetscFunctionBegin;
                PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
                bs   = mapping->bs;
                Nmax = bs*mapping->n;
                if (bs == 1) {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                } else {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                }
                //PetscFunctionReturn(0);
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
              //ierr = ISLocalToGlobalMappingApply(mat->cmap->mapping,ncol,icol,icolm);CHKERRQ(ierr);
              //PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
              //-- enter --
              {
                PetscInt i_,bs,Nmax;
                ISLocalToGlobalMapping mapping;
                PetscInt N;
                const PetscInt *in;
                PetscInt *out;
          
                mapping = mat->cmap->mapping;
                N = ncol;
                in = icol;
                out = icolm;
          
                PetscFunctionBegin;
                PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
                bs   = mapping->bs;
                Nmax = bs*mapping->n;
                if (bs == 1) {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=2
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=3
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=4
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=5
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                } else {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=2
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=3
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=4
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=5
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                }
                //PetscFunctionReturn(0);
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
             }
              //ierr = MatSetValues(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
              //PetscErrorCode MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
              //-- enter --
              {
                PetscErrorCode ierr;
                PetscInt m;
                const PetscInt *idxm;
                PetscInt n;
                const PetscInt *idxn;
                const PetscScalar *v;
              
                m = nrow;
                idxm = irowm;
                n = ncol;
                idxn = icolm;
                v = y;
              
                PetscFunctionBeginHot;
                PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
                PetscValidType(mat,1);
               if (m && n) { /* else : no values to insert */
                PetscValidIntPointer(idxm,3);
                PetscValidIntPointer(idxn,5);
                MatCheckPreallocated(mat,1);

                //ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
                //ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
                //PetscErrorCode MatSetValues_MPIAIJ(Mat mat,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode addv)
                //-- enter --
                {
                  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
                  PetscScalar    value = 0.0;
                  PetscErrorCode ierr;
                  PetscInt       i_,rstart  = mat->rmap->rstart,rend = mat->rmap->rend;
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
                  i_=0;
                  if (i_<m) { // max(m)=max(nrow)=1
                    if (im[i_*xm+i-xs] < 0) /* continue */ ; else
                    {
#ifdef VE_SETERRQ
                    if (PetscUnlikely(im[i_*xm+i-xs] >= mat->rmap->N)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",im[i_*xm+i-xs],mat->rmap->N-1);
#endif
                    if (im[i_*xm+i-xs] >= rstart && im[i_*xm+i-xs] < rend) {
                      row      = im[i_*xm+i-xs] - rstart;
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
                
                      if (jj<n) { // max(n)=max(ncol)=number of jj loop
                        if (v_tb)  value = roworiented ? v_tb[(i_*n+jj)*xm+i-xs] : v_tb[(i_+jj*m)*xm+i-xs];
                        if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES) && im[i_*xm+i-xs] != in[jj*xm+i-xs]) {
                          ; // continue;
                        } else
                        if (in[jj*xm+i-xs] >= cstart && in[jj*xm+i-xs] < cend) {
                          ; // never enter
                        } else if (in[jj*xm+i-xs] < 0) {;} // continue;
#ifdef VE_SETERRQ
                        else if (in[jj*xm+i-xs] >= mat->cmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[jj*xm+i-xs],mat->cmap->N-1);
#endif
                        else {
                          if (mat->was_assembled) {
#if defined(PETSC_USE_CTABLE)
                            ierr = PetscTableFind(aij->colmap,in[jj*xm+i-xs]+1,&col);CHKERRQ(ierr);
                            col--;
#else
                            col = aij->colmap[in[jj*xm+i-xs]] - 1;
#endif
                            if (col < 0 && !((Mat_SeqAIJ*)(aij->B->data))->nonew) {
                              ierr = MatDisAssemble_MPIAIJ(mat);CHKERRQ(ierr);
                              col  =  in[jj*xm+i-xs];
                              /* Reinitialize the variables required by MatSetValues_SeqAIJ_B_Private1() */
                              B        = aij->B;
                              b        = (Mat_SeqAIJ*)B->data;
                              bimax    = b->imax; bi = b->i; bilen = b->ilen; bj = b->j; ba = b->a;
                              rp2      = bj + bi[row];
                              ap2      = ba + bi[row];
                              rmax2    = bimax[row];
                              nrow2    = bilen[row];
                              low2     = 0;
                              high2    = nrow2;
                              bm       = aij->B->rmap->n;
                              ba       = b->a;
                              inserted = PETSC_FALSE;
                            } else if (col < 0 && !(ignorezeroentries && value == 0.0)) {
#ifdef VE_SETERRQ
                              if (1 == ((Mat_SeqAIJ*)(aij->B->data))->nonew) {
                                ierr = PetscInfo3(mat,"Skipping of insertion of new nonzero location in off-diagonal portion of matrix %g(%D,%D)\n",(double)PetscRealPart(value),im[i_*xm+i-xs],in[jj*xm+i-xs]);CHKERRQ(ierr);
                              } else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", im[i_*xm+i-xs], in[jj*xm+i-xs]);
#endif
                            }
                          } else col = in[jj*xm+i-xs];
                          nonew = b->nonew;
                          //MatSetValues_SeqAIJ_B_Private1(row,col,value,addv,im[i],in[j]);
                          //#define MatSetValues_SeqAIJ_B_Private(row,col,value,addv,orow,ocol)
                          { //-- enter --
                            PetscInt  orow,ocol;
                            PetscBool bflg, gflg; // bflg means break flag, gflg means goto flag
                            orow = im[i_*xm+i-xs];
                            ocol = in[jj*xm+i-xs];

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
                                bflg = gflg = PETSC_TRUE;
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
                                bflg = gflg = PETSC_TRUE;
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
                                bflg = gflg = PETSC_TRUE;
                              }
                              if (bflg==PETSC_FALSE) _i++;
                            }
                            if (gflg == PETSC_FALSE && value == 0.0 && ignorezeroentries) {low2 = 0; high2 = nrow2;gflg = PETSC_TRUE;}
                            if (gflg == PETSC_FALSE && nonew == 1) {low2 = 0; high2 = nrow2; gflg = PETSC_TRUE;}
                            if (gflg == PETSC_FALSE ) {
#ifdef VE_SETERRQ
                              if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at global row/column (%D, %D) into matrix", orow, ocol);
#endif
                              //MatSeqXAIJReallocateAIJ(B,bm,1,nrow2,row,col,rmax2,ba,bi,bj,rp2,ap2,bimax,nonew,MatScalar);
                              // if nrow2<rmax2 do nothing
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
                      if (mat->nooffprocentries) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Setting off process row %D even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set",im[i_*xm+i-xs]);
#endif
                      //if (!aij->donotstash) {
                      //  mat->assembled = PETSC_FALSE;
                      //  if (roworiented) {
                      //    ierr = MatStashValuesRow_Private(&mat->stash,im[i_],n,in,v+i*n,(PetscBool)(ignorezeroentries && (addv == ADD_VALUES)));CHKERRQ(ierr);
                      //  } else {
                      //    ierr = MatStashValuesCol_Private(&mat->stash,im[i_],n,in,v+i,m,(PetscBool)(ignorezeroentries && (addv == ADD_VALUES)));CHKERRQ(ierr);
                      //  }
                      //}
                    }
                    }
                    i_++;
                  }
                  //PetscFunctionReturn(0);
                }
                //-- exit --
                ierr = 0;CHKERRQ(ierr);
                //ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
                //PetscFunctionReturn(0);
               }
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
            //ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
            //PetscFunctionReturn(0);
           }
          }
          //-- exit
          ierr = 0;CHKERRQ(ierr);
          //PetscFunctionReturn(0);
         }
        }
        //-- exit --
        ierr = 0;CHKERRQ(ierr);
    }
   }
  }
 } else if (flg2==PETSC_TRUE && (1+5)*xm <= BUF_SIZE && !jac->ops->setvalueslocal && !PetscDefined(USE_DEBUG)) {
  for (j=ys; j<ys+ym; j++) {
   PetscInt k,l,low,high;
   for (l=0; l<MAX_CNT; l++) {
#pragma _NEC ivdep
    for (i=xs; i<xs+xm; i++) {
     if (l==0) {
      row_tb[i-xs].i = i; row_tb[i-xs].j = j;
      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num_tb[i-xs]=0; numi_tb[i-xs]=0; numj_tb[i-xs]=0;
        if (j!=0) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HxdHy; col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j-1;
          num_tb[i-xs]++; numj_tb[i-xs]++;
        }
        if (i!=0) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HydHx; col_tb[num_tb[i-xs]+5*(i-xs)].i = i-1; col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
          num_tb[i-xs]++; numi_tb[i-xs]++;
        }
        if (i!=M-1) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HydHx; col_tb[num_tb[i-xs]+5*(i-xs)].i = i+1; col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
          num_tb[i-xs]++; numi_tb[i-xs]++;
        }
        if (j!=N-1) {
          v_tb[num_tb[i-xs]*xm+i-xs] = -HxdHy; col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j+1;
          num_tb[i-xs]++; numj_tb[i-xs]++;
        }
        v_tb[num_tb[i-xs]*xm+i-xs] = ((PetscReal)(numj_tb[i-xs])*HxdHy + (PetscReal)(numi_tb[i-xs])*HydHx); col_tb[num_tb[i-xs]+5*(i-xs)].i = i;   col_tb[num_tb[i-xs]+5*(i-xs)].j = j;
        num_tb[i-xs]++;
        //ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        v_tb[0*xm+i-xs] = -HxdHy;              col_tb[0+5*(i-xs)].i = i;   col_tb[0+5*(i-xs)].j = j-1;
        v_tb[1*xm+i-xs] = -HydHx;              col_tb[1+5*(i-xs)].i = i-1; col_tb[1+5*(i-xs)].j = j;
        v_tb[2*xm+i-xs] = 2.0*(HxdHy + HydHx); col_tb[2+5*(i-xs)].i = i;   col_tb[2+5*(i-xs)].j = j;
        v_tb[3*xm+i-xs] = -HydHx;              col_tb[3+5*(i-xs)].i = i+1; col_tb[3+5*(i-xs)].j = j;
        v_tb[4*xm+i-xs] = -HxdHy;              col_tb[4+5*(i-xs)].i = i;   col_tb[4+5*(i-xs)].j = j+1;
        //ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
     }
        //PetscErrorCode MatSetValuesStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],const PetscScalar v[],InsertMode addv)
        //-- enter --
        {
          PetscErrorCode ierr;
          PetscInt m;
          Mat mat;
          const MatStencil *idxm;
          const MatStencil *idxn;
          mat = jac;
          idxm = &row_tb[i-xs];
          idxn = &col_tb[5*(i-xs)];
          PetscInt       *bufm=NULL,*bufn=NULL,*jdxm,*jdxn;
          PetscInt       j_,i_,dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
          PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

          PetscInt n;
          InsertMode addv;
        
          m = 1;
          if (i==0 || j==0 || i==M-1 || j==N-1) {
            n = num_tb[i-xs];
          } else {
            n = 5;
          }
          addv = INSERT_VALUES;
 
          PetscFunctionBegin;
         if (m && n) { /* else : no values to insert */
          PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
          PetscValidType(mat,1);
          PetscValidPointer(idxm,3);
          PetscValidPointer(idxn,5);
        
          jdxm = buf_tb1; jdxn = buf_tb1+m*xm;

         if (l==0) {
          i_=0;
          if (i_<m) { // max(m)=1
            j_=0;
            if (j_<3-sdim) { dxm++; j_++; } // max(3-sdim)=1
            tmp = *dxm++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxm++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxm-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxm++;
            jdxm[i_*xm+i-xs] = tmp;
            i_++;
          }
          i_=0;
          if (i_<n) { // max(n)=1
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=2
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=3
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=4
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
          if (i_<n) { // max(n)=5
            j_=0;
            if (j_<3-sdim) { dxn++; j_++; } // max(3-sdim)=1
            tmp = *dxn++ - starts[0];
            j_=0;
            if (j_<dim-1) { // max(dim-1)=1
              if ((*dxn++ - starts[j_+1]) < 0 || tmp < 0) tmp = -1;
              else                                       tmp = tmp*dims[j_] + *(dxn-1) - starts[j_+1];
              j_++;
            }
            if (mat->stencil.noc) dxn++;
            jdxn[i_*xm+i-xs] = tmp;
            i_++;
          }
         }
          //ierr = MatSetValuesLocal(mat,m,jdxm,n,jdxn,v,addv);CHKERRQ(ierr);
          //PetscErrorCode MatSetValuesLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
          //-- enter --
          {
            PetscErrorCode ierr;
            PetscInt nrow;
            const PetscInt *irow;
            PetscInt ncol;
            const PetscInt *icol;
            const PetscScalar *y;
          
            nrow = m;
            irow = jdxm;
            ncol = n;
            icol = jdxn;
            y = v;
          
            PetscFunctionBeginHot;
            PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
            PetscValidType(mat,1);
            MatCheckPreallocated(mat,1);
           if (nrow && ncol) { /* else : no values to insert */
            PetscValidIntPointer(irow,3);
            PetscValidIntPointer(icol,5);
#ifdef VE_SETERRQ
            else if (PetscUnlikely(mat->insertmode != addv)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot mix add values and insert values");
            if (PetscDefined(USE_DEBUG)) {
              if (mat->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
              if (!mat->ops->setvalueslocal && !mat->ops->setvalues) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);
            }
#endif
              PetscInt *bufr=NULL,*bufc=NULL,*irowm,*icolm;
              irowm = buf_tb2; icolm = buf_tb2+nrow*xm;
#ifdef VE_SETERRQ
              if (!mat->rmap->mapping) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatSetValuesLocal() cannot proceed without local-to-global row mapping (See MatSetLocalToGlobalMapping()).");
              if (!mat->cmap->mapping) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"MatSetValuesLocal() cannot proceed without local-to-global column mapping (See MatSetLocalToGlobalMapping()).");
#endif
             if (l==0) {
              //ierr = ISLocalToGlobalMappingApply(mat->rmap->mapping,nrow,irow,irowm);CHKERRQ(ierr);
              //PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
              //-- enter --
              {
                PetscInt i_,bs,Nmax;
                ISLocalToGlobalMapping mapping;
                PetscInt N;
                const PetscInt *in;
                PetscInt *out;
          
                mapping = mat->rmap->mapping;
                N = nrow;
                in = irow;
                out = irowm;
          
                PetscFunctionBegin;
                PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
                bs   = mapping->bs;
                Nmax = bs*mapping->n;
                if (bs == 1) {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                } else {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                }
                //PetscFunctionReturn(0);
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
              //ierr = ISLocalToGlobalMappingApply(mat->cmap->mapping,ncol,icol,icolm);CHKERRQ(ierr);
              //PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
              //-- enter --
              {
                PetscInt i_,bs,Nmax;
                ISLocalToGlobalMapping mapping;
                PetscInt N;
                const PetscInt *in;
                PetscInt *out;
          
                mapping = mat->cmap->mapping;
                N = ncol;
                in = icol;
                out = icolm;
          
                PetscFunctionBegin;
                PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
                bs   = mapping->bs;
                Nmax = bs*mapping->n;
                if (bs == 1) {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=2
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*(xs+xm)+i);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=3
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=4
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=5
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]];
                    }
                    i_++;
                  }
                } else {
                  const PetscInt *idx = mapping->indices;
                  i_=0;
                  if (i_<N) { // max(N)=1
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=2
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=3
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=4
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                  if (i_<N) { // max(N)=5
                    if (in[i_*xm+i-xs] < 0) {
                      out[i_*xm+i-xs] = in[i_*xm+i-xs];
                    } else {
#ifdef VE_SETERRQ
                      if (in[i_*xm+i-xs] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i_*xm+i-xs],Nmax-1,i_*xm+i-xs);
#endif
                      out[i_*xm+i-xs] = idx[in[i_*xm+i-xs]/bs]*bs + (in[i_*xm+i-xs] % bs);
                    }
                    i_++;
                  }
                }
                //PetscFunctionReturn(0);
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
             }
              //ierr = MatSetValues(mat,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
              //PetscErrorCode MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
              //-- enter --
              {
                PetscErrorCode ierr;
                PetscInt m;
                const PetscInt *idxm;
                PetscInt n;
                const PetscInt *idxn;
                const PetscScalar *v;
              
                m = nrow;
                idxm = irowm;
                n = ncol;
                idxn = icolm;
                v = y;
              
                PetscFunctionBeginHot;
                PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
                PetscValidType(mat,1);
               if (m && n) { /* else : no values to insert */
                PetscValidIntPointer(idxm,3);
                PetscValidIntPointer(idxn,5);
                MatCheckPreallocated(mat,1);

                //ierr = PetscLogEventBegin(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
                //ierr = (*mat->ops->setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
                //PetscErrorCode MatSetValues_SeqAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
                //-- enter --
                {
                  Mat A;
                  A = mat;
                  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
                  PetscInt       *rp,t,ii,row,nrow,col,rmax,N;
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
                    row = im[k*xm+i-xs];
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
                     if (in[l*xm+i-xs] >= 0) { // if in[] < 0 continue;
#ifdef VE_SETERRQ
                      if (PetscUnlikelyDebug(in[l*xm+i-xs] >= A->cmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l*xm+i-xs],A->cmap->n-1);
#endif
                      col = in[l*xm+i-xs];
                      if (v_tb && !A->structure_only) value = roworiented ? v_tb[(l + k*n)*xm+i-xs] : v_tb[(k + l*m)*xm+i-xs];
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
                //-- exit --
                ierr = 0;CHKERRQ(ierr);
                //ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
                //PetscFunctionReturn(0);
               }
              }
              //-- exit --
              ierr = 0;CHKERRQ(ierr);
            //ierr = PetscLogEventEnd(MAT_SetValues,mat,0,0,0);CHKERRQ(ierr);
            //PetscFunctionReturn(0);
           }
          }
          //-- exit
          ierr = 0;CHKERRQ(ierr);
          //PetscFunctionReturn(0);
         }
        }
        //-- exit --
        ierr = 0;CHKERRQ(ierr);
    } // i loop
   } // l loop
  } // j loop 
 } else {
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num=0; numi=0; numj=0;
        if (j!=0) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j-1;
          num++; numj++;
        }
        if (i!=0) {
          v[num] = -HydHx;              col[num].i = i-1; col[num].j = j;
          num++; numi++;
        }
        if (i!=M-1) {
          v[num] = -HydHx;              col[num].i = i+1; col[num].j = j;
          num++; numi++;
        }
        if (j!=N-1) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j+1;
          num++; numj++;
        }
        v[num] = ((PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx); col[num].i = i;   col[num].j = j;
        num++;
        ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
 }
#else
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num=0; numi=0; numj=0;
        if (j!=0) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j-1;
          num++; numj++;
        }
        if (i!=0) {
          v[num] = -HydHx;              col[num].i = i-1; col[num].j = j;
          num++; numi++;
        }
        if (i!=M-1) {
          v[num] = -HydHx;              col[num].i = i+1; col[num].j = j;
          num++; numi++;
        }
        if (j!=N-1) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j+1;
          num++; numj++;
        }
        v[num] = ((PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx); col[num].i = i;   col[num].j = j;
        num++;
        ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
#endif
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type svd -ksp_view

   test:
      suffix: 2
      nsize: 4
      args: -pc_type mg -pc_mg_type full -ksp_type cg -ksp_monitor_short -da_refine 3 -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type svd -ksp_view

   test:
      suffix: 3
      nsize: 2
      args: -pc_type mg -pc_mg_type full -ksp_monitor_short -da_refine 5 -mg_coarse_ksp_type cg -mg_coarse_ksp_converged_reason -mg_coarse_ksp_rtol 1e-2 -mg_coarse_ksp_max_it 5 -mg_coarse_pc_type none -pc_mg_levels 2 -ksp_type pipefgmres -ksp_pipefgmres_shift 1.5

   test:
      suffix: tut_1
      nsize: 1
      args: -da_grid_x 4 -da_grid_y 4 -mat_view

   test:
      suffix: tut_2
      requires: superlu_dist parmetis
      nsize: 4
      args: -da_grid_x 120 -da_grid_y 120 -pc_type lu -pc_factor_mat_solver_type superlu_dist -ksp_monitor -ksp_view

   test:
      suffix: tut_3
      nsize: 4
      args: -da_grid_x 1025 -da_grid_y 1025 -pc_type mg -pc_mg_levels 9 -ksp_monitor

TEST*/
