#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=regular-a"
#PJM -L "elapse=10:00"
#PJM -s
#PJM -g gi37
#PJM --mpi proc=16

module load nvidia/21.3
module load ompi-cuda/4.1.1-11.2
module list

export UCX_MEMTYPE_CACHE=n
export UCX_IB_GPU_DIRECT_RDMA=n
export UCX_MAX_RNDV_RAILS=4
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export OMPI_MCA_mtl=^ofi
export OMPI_MCA_btl=^uct

mpiexec -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -npernode 8 \
    ./vlp4d.WISTERIA_A100_openacc --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f SLD10.dat
