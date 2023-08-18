#!/bin/sh

export NMPI_PROGINF=DETAIL
export VE_PROGINF=DETAIL

grid_num=1025
EXE=src/ksp/ksp/tutorials/ksp_MPI_ex50
OPTION="-da_grid_x "${grid_num}" -da_grid_y "${grid_num}" -pc_type mg -pc_mg_levels 9 -ksp_monitor -log_view"

for mpi_num in 1 2 4 8 ; do
  echo ${mpi_num}
  LOG=ex50_${grid_num}_${grid_num}_${mpi_num}.MPI_static.res
  ERR=ex50_${grid_num}_${grid_num}_${mpi_num}.MPI_static.err
  mpiexec -np ${mpi_num} ${EXE} ${OPTION} 1>${LOG} 2> ${ERR}
done
