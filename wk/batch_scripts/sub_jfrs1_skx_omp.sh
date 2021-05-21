#!/bin/sh
#SBATCH -J omp               # jobname
#SBATCH --nodes=4            # Number of nodes
#SBATCH --ntasks-per-node=4  # Number of processes per node
#SBATCH --cpus-per-task=20   # logical core per task
#SBATCH --time=00:30:00      # execute time (hh:mm:ss)
#SBATCH --account=GYS_DATA   # account number
#SBATCH -o %j.out            # strout filename (%j is jobid)
#SBATCH -e %j.err            # stderr filename (%j is jobid)
#SBATCH -p S-M               # Job class

source /opt/modules/default/init/bash
module switch PrgEnv-intel PrgEnv-gnu
module unload cray-libsci/18.04.1
module load intel
module load cray-fftw

export OMP_NUM_THREADS=20
export OMP_STACKSIZE=16M # To increase the stacksize
export OMP_PROC_BIND=true
srun ./vlp4d.jfrs1_skx_openmp --num_threads 20 --teams 1 -f SLD10.dat
