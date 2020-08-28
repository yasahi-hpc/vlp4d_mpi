#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscunit=fx"
#PJM -L "rscgrp=fx-small"
#PJM -L "elapse=10:00"
#PJM --mpi "proc=8"
#PJM -s
                  
module list
module load tcs/1.2.25
module load fftw-tune
                       
export OMP_NUM_THREADS=12
export OMPI_MCA_plm_ple_memory_allocation_policy=bind_local
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

mpiexec ./vlp4d.flow_a64fx_openmp --num_threads 12 --teams 1 -f SLD10.dat
