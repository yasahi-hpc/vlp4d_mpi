#!/bin/bash
#PBS -q pg9
#PBS -l select=2:ncpus=48:mpiprocs=4:ompthreads=12:ngpus=4
#PBS -l walltime=00:10:00
#PBS -P vlp4d

cd $PBS_O_WORKDIR
. /etc/profile.d/modules.sh

module purge
module load cuda/11.0
module load pgi/20.9
module load mpt/2.23-ga

export MPI_USE_CUDA=1

mpirun -np 8 ./vlp4d.s86_v100_openacc --num_threads 1 --teams 1 --device 0 --num_gpus 4 --device_map 1 -f SLD10.dat
