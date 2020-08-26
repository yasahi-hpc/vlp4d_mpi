#!/bin/sh
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -L rscgrp=regular-cache
#PJM -g hp150279
#PJM --mpi proc=4
#PJM --omp thread=16

source /usr/local/bin/hybrid_core_setting.sh 2

module purge
module load intel
module load impi
module load fftw

export OMP_NUM_THREADS=16

mpiexec.hydra -ppn 4 \
              -n 4 \
              ./vlp4d.pacs_knl_openmp --num_threads 16 --teams 1 -f SLD10.dat
