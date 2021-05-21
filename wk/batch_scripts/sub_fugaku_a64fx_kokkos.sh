#!/bin/bash
#PJM -L "node=4"
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=eap-small"
#PJM -L "elapse=10:00"
#PJM --mpi "proc=16"
#PJM -s

module list
module load lang/tcsds-1.2.31

export OMP_NUM_THREADS=12
export OMPI_MCA_plm_ple_memory_allocation_policy=bind_local
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

mpiexec ./vlp4d.flow_a64fx_kokkos --num_threads 12 --teams 1 -f SLD10.dat
