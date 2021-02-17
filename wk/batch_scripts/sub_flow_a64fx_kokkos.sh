#!/bin/bash
#PJM -L "node=4"
#PJM -L "rscunit=fx"
#PJM -L "rscgrp=fx-small"
#PJM -L "elapse=3:00:00"
#PJM --mpi "proc=16"
#PJM -s
                  
module list
module load tcs/1.2.27
module load fftw-tune
                       
export OMP_NUM_THREADS=12
export OMPI_MCA_plm_ple_memory_allocation_policy=bind_local
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

mpiexec ./vlp4d.flow_a64fx_kokkos --num_threads 12 --teams 1 -f SLD10.dat
