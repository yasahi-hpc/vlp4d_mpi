# About

The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. Vlasov solver is typically based on a directional Strang splitting. The Poisson equation is treated with 2D Fourier transforms. For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions. As a major update from [the pervious version](https://github.com/yasahi-hpc/vlp4d), we parallelized the code with MPI and upgrade the interpolatin scheme from Lagrange to Spline. 

The Vlasov solver is based on advection's operators: 
- Halo excahnge on f^n (P2P communications)  
- Compute spline coefficient along (x, y)
- 2D advection along x, y (Dt/2)
- Poisson solver -> compute electric fields Ex and E
- Compute spline coefficient along (vx, vy)
- 4D advection along x, y, vx, vy directions for Dt

Detailed descriptions of the test cases can be found in 
- [Crouseilles & al. J. Comput. Phys., 228, pp. 1429-1446, (2009).](http://people.rennes.inria.fr/Nicolas.Crouseilles/loss4D.pdf)  
  Section 5.3.1 Two-dimensional Landau damping -> SLD10
- [Crouseilles & al. Communications in Nonlinear Science and Numerical Simulation, pp 94-99, 13, (2008).](http://people.rennes.inria.fr/Nicolas.Crouseilles/cgls2.pdf)  
  Section 2 and 3 Two stream Instability and Beam focusing pb -> TSI20
- [Crouseilles & al. Beam Dynamics Newsletter no 41 (2006).](http://icfa-bd.kek.jp/Newsletter41.pdf )  
  Section 3.3, Beam focusing pb.
  
For questions or comments, please find us in the AUTHORS file.

# HPC
From the view point of high perfomrance computing (HPC), the code is parallelized with OpenMP without MPI domain decomposition.
In order to investigate the performance portability of this kind of kinietic plasma simulation codes, we implement the mini-app with
a mixed OpenACC/OpenMP and Kokkos, where we suppress unnecessary duplications of code lines. The detailed description and obtained performance is found in
- [Yuuichi Asahi, Guillaume Latu, Virginie Grandgirard, and Julien Bigot, "Performance Portable Implementation of a Kinetic Plasma Simulation Mini-app"](https://sc19.supercomputing.org/proceedings/workshops/workshop_files/ws_waccpd104s2-file1.pdf), in Proceedings of Sixth Workshop on Accelerator Programming Using Directives (WACCPD), IEEE, 2019.

# Test environments
We have tested the code on the following environments. 
- Nvidia Tesla p100 on Tsubame3.0 (Tokyo Tech, Japan)  
Compilers: cuda/10.2.48 + openmpi3.1.4 (Kokkos), pgi19.1 + openmpi3.1.4 (OpenACC)

- Nvidia Tesla v100 on Marconi100 (Cineca, Italy)  
Compilers cuda/10.2 + spectrum_mpi10.3.1 (Kokkos), Nvidia HPC SDK 20.11-0 (OpenACC)

- Intel Skylake on JFRS-1 (IFERC-CSC, Japan)  
Compilers (intel compiler 19.0.0.117)

- Fujitsu A64FX on Flow (Nagoya Univ., Japan)  
Compilers (Fujitsu compiler 1.2.27)

# Usage
## Compile
Firstly, you need to git clone on your environment as
```
git clone https://github.com/yasahi-hpc/vlp4d_mpi.git
```
Depending on your configuration, you may have to modify the Makefile.
You may add your configuration in the same way as 
```
ifneq (,$(findstring p100,$(DEVICES)))
CXX      = mpicxx
CXXFLAGS = -O3 -ta=nvidia:cc60 -Minfo=accel -Mcudalib=cufft,cublas -std=c++11 -DENABLE_OPENACC -DLAYOUT_LEFT
LDFLAGS  = -Mcudalib=cufft,cublas -ta=nvidia:cc60 -acc
TARGET   = vlp4d.tsubame3.0_p100_openacc
endif
```
Before compiling, you need to load appropriate modules for MPI + CUDA/OpenACC environment. 
The CUDA-Aware-MPI is necessary for this application.
For CPU version, it is also necessary to make sure that [fftw](http://www.fftw.org) is avilable in your configuration. 

### OpenACC version
```
export DEVICE=device_name # choose the device_name from "p100", "bdw", "knl", "skx", "a64fx_flow"
cd src_openacc
make
```

### Kokkos version
First of all, you need to install kokkos on your environment. Instructions are found in https://github.com/kokkos/kokkos. In the following example, it is assumed that kokkos is located at "your_kokkos_path".

```
export KOKKOS_PATH=your_kokkos_path # set your_kokkos_path
export OMPICXX=$KOKKOSPATH/bin/nvccwrapper # Assuming OpenMPI as a MPI library (GPU only)
export DEVICE=device_name # choose the device_name from "p100", "bdw", "skx", "fugaku_a64fx", "flow_a64fx"
cd src_kokkos
make
```

## Run
Depending on your configuration, you may have to modify the job.sh in wk and sub_*.sh in wk/batch_scripts.

```
cd wk
./job.sh
```

You can also try the two beam instability by setting the argument as "TSI20.dat".

## Experimental workflow
In order to evaluate the impact of optimizations, one has to compile the code on each environment. 
Here, ```device_name``` is the name of the device one use in Makefile and job scripts. We assume that the Installation has already been made successfully. The impact of optimizations can be evaluated by comparing the standard output with different versions.

### OpenACC version
```
export DEVICE=device_name
export OPTIMIZATION=STEP1 # choose from STEP0-2 for CPUs and choose from STEP0-1 for GPUs
cd src_openacc
make
cd ../wk
./job.sh
```

### Kokkos version
```
export DEVICE=device_name
export OMPICXX=$KOKKOSPATH/bin/nvccwrapper # Only for OpenMPI + GPU case
export OPTIMIZATION=STEP1 # choose from STEP0-2 for CPUs and choose from STEP0-1 for GPUs
cd src_kokkos
make
cd ../wk
./job.sh
```

### Expected result
If the code works correctly, one may find the standard output file in ascii format showing the timing at the bottom.  
The timings look like (though not alingned in the standard output file)

|  |  |  | 
| ---- | ---- | ---- | 
| total | 4.57123 [s], | 1 calls |  
| MainLoop | 4.56027 [s], | 40 calls |
| pack | 0.14395 [s], | 40 calls |
| comm | 0.705633 [s], | 40 calls |
| unpack | 0.0556184 [s], | 40 calls | 
| advec2D | 0.258894 [s], | 40 calls |
| advec4D |1.38773 [s], | 40 calls |
| field |0.0474864 [s], | 80 calls |
| all\_reduce |0.116563 [s], | 80 calls |
| Fourier |0.0296469 [s], | 80 calls |
| diag |0.0992476 [s], | 40 calls |
| splinecoeff\_xy | 0.805345 [s], | 40 calls |
| splinecoeff\_vxvy | 0.907955 [s], | 40 calls |

Each column denotes the kernel name, total elapsed time in seconds, and number of call counts.
The elapsed time ```s``` of a given kernel for a single iteration can be computed by
```
elapsed time [s] / number of call counts
```

The Flops and memory bandwidth are computed by the following formula
```
Flops = Nf/s,
Bytes/s = Nb/s
```
where ```f``` and ```b``` denote the total amount of floating point operation and memory accesses per grid point. ```N``` represent the total number of grid points ans ```s``` is the elapsed time of a given kernel for a single iteration. ```f``` and ```b``` presented in Table 3 are the analytical estimates from the source code.
