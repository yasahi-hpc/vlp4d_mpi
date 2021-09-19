#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g gi37
#PJM --mpi proc=16

module purge
module load nvidia/21.3 cuda/11.2 ompi-cuda/4.1.1-11.2
module list

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=n

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ./batch_scripts/wrapper.sh ./vlp4d.WISTERIA_A100_omp4.5 --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f SLD10.dat
