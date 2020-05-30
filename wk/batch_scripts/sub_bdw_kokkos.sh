#!/bin/sh
#$ -cwd               # job execution in the current directory
#$ -l f_node=1        # Using one f_node
#$ -l h_rt=0:10:00    # Execution time
#$ -N parallel
. /etc/profile.d/modules.sh # Initialize module command
module load cuda intel
module load openmpi/2.1.2-opa10.9-t3-thread-multiple

export OMP_NUM_THREADS=14
export OMP_PROC_BIND=true
mpirun -npernode 2 -np 2 ./vlp4d.bdw_kokkos --num_threads 14 --teams 1 -f SLD10.dat
