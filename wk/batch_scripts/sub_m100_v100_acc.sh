#!/bin/bash
#SBATCH -A FUSIO_jaPP-GYS
#SBATCH -p m100_fua_prod
#SBATCH --time 00:10:00     # format: HH:MM:SS
#SBATCH --nodes=4           # 1 node
#SBATCH --ntasks-per-node=4 # 2 MPI processes per node
#SBATCH --cpus-per-task=32  
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --mem=230000MB
#SBATCH --job-name=Vlp4d
#SBATCH --exclusive
#SBATCH --gpu-bind=closest

module purge
module load cuda/10.2
module load hpc-sdk/2020--binary
mpirun ./vlp4d.m100_v100_openacc --num_threads 1 --teams 1 --device 0 --num_gpus 4 --device_map 1 -f SLD10.dat
