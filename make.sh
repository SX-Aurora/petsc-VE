#!/bin/sh

COM_VER=5.0.21

./configure \
            --with-batch=true \
            --with-cc=${NMPI_ROOT}/bin/mpicc \
            --CFLAGS="-O3 -fdiag-vector=2 -fdiag-inline=2 -fdiag-parallel=2 -finline-functions" \
            --with-cxx=${NMPI_ROOT}/bin/mpic++ \
            --CXXFLAGS="-O3 -fdiag-vector=2 -fdiag-inline=2 -fdiag-parallel=2 -finline-functions" \
            --with-fc=${NMPI_ROOT}/bin/mpifort \
            --FFLAGS="-O3 -fdiag-vector=2 -fdiag-inline=2 -fdiag-parallel=2" \
            --with-fortranlib-autodetect=0 \
            --with-debugging=0 \
            --with-shared-libraries=0 \
            --with-mpi=1 \
            --with-mpi-include=${NMPI_ROOT}/include \
            --with-mpi-lib=${NMPI_ROOT}/lib/ve/libpmpi.a \
            --with-mpiexec=mpiexec \
            --with-mpi-compilers=1 \
            --with-blas-lib=${NLC_HOME}/lib/libblas_sequential.a \
            --with-lapack-lib=${NLC_HOME}/lib/liblapack.a \
            --with-petsc-arch="Aurora_MPI_static" \
            --CC_LINKER_FLAGS="/opt/nec/ve/nfort/${COM_VER}/lib/nousemmap.o /opt/nec/ve/nfort/${COM_VER}/lib/quickfit.o /opt/nec/ve/nfort/${COM_VER}/lib/async_noio.o -Wl,-rpath,/opt/nec/ve/nfort/${COM_VER}/lib -L/opt/nec/ve/nfort/${COM_VER}/lib -lnfort" \
            --CXX_LINKER_FLAGS="/opt/nec/ve/nfort/${COM_VER}/lib/nousemmap.o /opt/nec/ve/nfort/${COM_VER}/lib/quickfit.o /opt/nec/ve/nfort/${COM_VER}/lib/async_noio.o -Wl,-rpath,/opt/nec/ve/nfort/${COM_VER}/lib -L/opt/nec/ve/nfort/${COM_VER}/lib -lnfort"

make PETSC_DIR=./ PETSC_ARCH=Aurora_MPI_static all

TP=src/ksp/ksp/tutorials/ex50

PETSC_DIR=./
PETSC_ARCH=Aurora_MPI_static
PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib
PETSC_INCLUDE=$PETSC_DIR/$PETSC_ARCH/include

BLAS_LAPACK_LIB=${NLC_HOME}/lib/

${NMPI_ROOT}/bin/mpinfort \
-I$PETSC_DIR/include/ \
-I$PETSC_INCLUDE \
$TP.c \
-L $PETSC_LIB -lpetsc \
-L $BLAS_LAPACK_LIB -llapack \
-L $BLAS_LAPACK_LIB -lblas_sequential

mv ex50.o src/ksp/ksp/tutorials
mv a.out src/ksp/ksp/tutorials/ksp_MPI_ex50
