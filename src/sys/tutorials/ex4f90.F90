!
!     This introductory example illustrates running PETSc on a subset
!     of processes
!
!/*T
!   Concepts: introduction to PETSc;
!   Concepts: process^subset set PETSC_COMM_WORLD
!   Processors: 2
!T*/
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscsys.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscsys
      implicit none

      PetscErrorCode ierr
      PetscMPIInt rank, size, zero, two

!     We must call MPI_Init() first, making us, not PETSc, responsible
!     for MPI

      call MPI_Init(ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize MPI'
         stop
      endif

!     We can now change the communicator universe for PETSc

      zero = 0
      two = 2
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      call MPI_Comm_split(MPI_COMM_WORLD,mod(rank,two),zero,PETSC_COMM_WORLD,ierr)

!     Every PETSc routine should begin with the PetscInitialize()
!     routine.

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

!     The following MPI calls return the number of processes being used
!     and the rank of this process in the group.

      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr);CHKERRA(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)

!     Here we would like to print only one message that represents all
!     the processes in the group.
      if (rank .eq. 0) write(6,100) size,rank
 100  format('No of Procs = ',i4,' rank = ',i4)

!     Always call PetscFinalize() before exiting a program.  This
!     routine - finalizes the PETSc libraries as well as MPI - provides
!     summary and diagnostic information if certain runtime options are
!     chosen (e.g., -log_view).  See PetscFinalize() manpage for more
!     information.

      call PetscFinalize(ierr)
      call MPI_Comm_free(PETSC_COMM_WORLD,ierr)

!     Since we initialized MPI, we must call MPI_Finalize()

      call  MPI_Finalize(ierr)
      end

!
!/*TEST
!
!   test:
!
!TEST*/
