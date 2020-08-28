#!/bin/sh
#$ -cwd               # job execution in the current directory
#$ -l f_node=2        # Using one f_node
#$ -l h_rt=0:10:00    # Execution time
#$ -N parallel
. /etc/profile.d/modules.sh # Initialize module command
module load cuda intel
module load openmpi/3.1.4-opa10.10-t3

#export OMP_PROC_BIND=true
export OMP_NUM_THREADS=7
mpirun --mca pml ob1 -mca mtl psm2 -npernode 4 -np 8 -x PSM2_CUDA=1 -x PSM2_GPUDIRECT=1 -x LD_LIBRARY_PATH -x PATH \
       -x CUDA_VISIBLE_DEVICES=0,1,2,3 -x HFI_UNIT=0,1,2,3 -x PSM2_MULTIRAIL=2 \
       --bind-to board ./vlp4d.tsubame3.0_p100_kokkos --num_threads 1 --teams 1 --device 0 --num_gpus 4 --device_map 1 -f SLD10.dat
