#!/bin/sh
#$ -cwd               # job execution in the current directory
#$ -l f_node=1        # Using one f_node
#$ -l h_rt=0:10:00    # Execution time
#$ -N parallel
. /etc/profile.d/modules.sh # Initialize module command
module load cuda intel
module load openmpi/2.1.2-opa10.9-t3-thread-multiple

export OMP_PROC_BIND=true
mpirun --mca pml ob1 -mca mtl psm2 -npernode 2 -np 2 -x PSM2_CUDA=1 -x PSM2_GPUDIRECT=1 -x LD_LIBRARY_PATH -x PATH ./vlp4d.p100_kokkos --num_threads 1 --teams 1 --device 0 --num_gpus 2 --device_map 1 -f SLD10.dat
