#!/bin/sh
#$ -cwd               # job execution in the current directory
#$ -l f_node=2        # Using one f_node
#$ -l h_rt=0:10:00    # Execution time
#$ -N parallel
. /etc/profile.d/modules.sh # Initialize module command
module load cuda intel
module load openmpi/3.1.4-opa10.10-t3
module load fftw

export OMP_NUM_THREADS=7
mpirun -npernode 4 -np 8 -x PATH -x LD_LIBRARY_PATH ./vlp4d.tsubame3.0_bdw_openmp --num_threads 7 --teams 1 -f SLD10.dat
