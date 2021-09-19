#!/bin/bash
#SBATCH -A FUSIO_ja5PPGYS
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
module load hpc-sdk/2021--binary

mpirun ./batch_scripts/wrapper.sh ./vlp4d.m100_v100_omp4.5 --num_threads 1 --teams 1 --device 0 --num_gpus 4 --device_map 1 -f SLD10.dat
