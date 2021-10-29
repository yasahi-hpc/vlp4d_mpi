# About

The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. Vlasov solver is typically based on a directional Strang splitting. The Poisson equation is treated with 2D Fourier transforms. For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions. As a major update from [the pervious version](https://github.com/yasahi-hpc/vlp4d), we parallelized the code with MPI and upgrade the interpolatin scheme from Lagrange to Spline. 

The Vlasov solver is based on advection's operators: 
- Halo excahnge on <img src="https://render.githubusercontent.com/render/math?math={f^{n}}"> (P2P communications)  
- Compute spline coefficient along <img src="https://render.githubusercontent.com/render/math?math={ \left(x, y \right)}">
- 2D advection along <img src="https://render.githubusercontent.com/render/math?math={x, y\ \left(\Delta t/2 \right) }">
- Poisson solver -> compute electric fields <img src="https://render.githubusercontent.com/render/math?math={E_x}"> and <img src="https://render.githubusercontent.com/render/math?math={E_y}">
- Compute spline coefficient along <img src="https://render.githubusercontent.com/render/math?math={ \left(v_{x}, v_{y} \right)}">
- 4D advection along <img src="https://render.githubusercontent.com/render/math?math={ \left(x, y, v_{x}, v_{y} \right)}"> directions for <img src="https://render.githubusercontent.com/render/math?math={\Delta t}">

Detailed descriptions of the test cases can be found in 
- [Crouseilles & al. J. Comput. Phys., 228, pp. 1429-1446, (2009).](http://people.rennes.inria.fr/Nicolas.Crouseilles/loss4D.pdf)  
  Section 5.3.1 Two-dimensional Landau damping -> SLD10
- [Crouseilles & al. Communications in Nonlinear Science and Numerical Simulation, pp 94-99, 13, (2008).](http://people.rennes.inria.fr/Nicolas.Crouseilles/cgls2.pdf)  
  Section 2 and 3 Two stream Instability and Beam focusing pb -> TSI20
- [Crouseilles & al. Beam Dynamics Newsletter no 41 (2006).](http://icfa-bd.kek.jp/Newsletter41.pdf )  
  Section 3.3, Beam focusing pb.
  
For questions or comments, please find us in the AUTHORS file.

# HPC
From the view point of high perfomrance computing (HPC), the code is parallelized with MPI + "X", where "X" is one of a mixed OpenMP3.0/OpenACC, OpenMP3.0/OpenMP4.5, Kokkos and parallel algorithm (experimental). We have investigated optimization strategies applicable to a kinetic plasma simulation code that makes use of the MPI + "X" implementation listed above. The details are presented in the [P3HPC workshop 2021](https://p3hpc.org/workshop/2021/). Our previous result for [non-MPI version](https://github.com/yasahi-hpc/vlp4d) is found in 
- [Yuuichi Asahi, Guillaume Latu, Virginie Grandgirard, and Julien Bigot, "Performance Portable Implementation of a Kinetic Plasma Simulation Mini-app"](https://link.springer.com/chapter/10.1007/978-3-030-49943-3_6), in [Accelerator Programming Using Directives](https://link.springer.com/book/10.1007/978-3-030-49943-3) or in [Proceedings of Sixth Workshop on Accelerator Programming Using Directives (WACCPD), IEEE, 2019](https://sc19.supercomputing.org/proceedings/workshops/workshop_files/ws_waccpd104s2-file1.pdf).

# Test environments
We have tested the code on the following environments. 
- Nvidia Tesla p100 on Tsubame3.0 (Tokyo Tech, Japan)  
Compilers: cuda/10.2.48 + openmpi3.1.4 (Kokkos), pgi19.1 + openmpi3.1.4 (OpenACC)

- Nvidia Tesla v100 on Marconi100 (Cineca, Italy)  
Compilers cuda/10.2 + spectrum_mpi10.3.1 (Kokkos), Nvidia HPC SDK 20.11-0 (OpenACC)

- Intel Skylake on JFRS-1 (IFERC-CSC, Japan)  
Compilers (intel compiler 18.0.2)

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
Before compiling, you need to load appropriate modules for MPI + CUDA/OpenACC/OpenMP4.5 environment. 
CUDA-Aware-MPI is necessary for this application.
For CPU version, it is also necessary to make sure that [fftw](http://www.fftw.org) is available in your configuration. 
OpenMP4.5 and stdpar versions are experimental and not appeared in the workshop paper.
For OpenMP4.5 and stdpar versions, we have only tested with ```nvc++``` in Nvidia HPC SDK.

### OpenACC version
```
export DEVICE=device_name # choose the device_name from "p100", "v100", "a100", "bdw", "knl", "skx", "a64fx"
cd src_openacc
make
```

### OpenMP4.5 version
This is an experimental version (not appeared in the workshop paper). 
```
export DEVICE=device_name # choose the device_name from "v100", "a100"
cd src_openmp4.5
make
```

### Kokkos version
First of all, you need to install kokkos on your environment. Instructions are found in https://github.com/kokkos/kokkos. In the following example, it is assumed that kokkos is located at "your_kokkos_path".

```
export KOKKOS_PATH=your_kokkos_path # set your_kokkos_path
export OMPICXX=$KOKKOSPATH/bin/nvccwrapper # Assuming OpenMPI as a MPI library (GPU only)
export DEVICE=device_name # choose the device_name from "p100", "v100", "a100", "bdw", "skx", "a64fx"
cd src_kokkos
make
```

### C++ parallel algorithm (stdpar) version
This is an experimental version (not appeared in the workshop paper). Performance test has been made on A100 GPU. Further optimizations are needed for this version.
```
export DEVICE=device_name # choose the device_name from "p100", "v100", "a100", "icelake"
cd src_stdpar
make
```

## Run
Depending on your configuration, you may have to modify the ```job.sh``` in ```wk``` and ```sub_*.sh``` in ```wk/batch_scripts```.

```
cd wk
./job.sh
```

## Experiment workflow
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

### OpenMP4.5 version
This is an experimental version (not appeared in the workshop paper). 
It seems important to map GPUs before calling ```MPI_Init```. See [wrapper.sh](https://github.com/yasahi-hpc/vlp4d_mpi/blob/master/wk/batch_scripts/wrapper.sh) and [sub_Wisteria_A100_omp4.5.sh](https://github.com/yasahi-hpc/vlp4d_mpi/blob/master/wk/batch_scripts/sub_Wisteria_A100_omp4.5.sh).
```
export DEVICE=device_name
export OPTIMIZATION=STEP1 # choose from STEP0-1 for GPUs
cd src_openmp4.5
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

### C++ parallel algorithm (stdpar) version
This is an experimental version (not appeared in the workshop paper). 
As well as the OpenMP4.5 version, it is recommended to map GPUs before calling ```MPI_Init```. See [sub_Wisteria_A100_stdpar.sh](https://github.com/yasahi-hpc/vlp4d_mpi/blob/master/wk/batch_scripts/sub_Wisteria_A100_stdpar.sh).

```
export DEVICE=device_name
cd src_stdpar
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
total elapsed time [s] / number of call counts
```

The Flops and memory bandwidth are computed by the following formula
```
Flops = Nf/s,
Bytes/s = Nb/s
```
where ```f``` and ```b``` denote the total amount of floating point operation and memory accesses per grid point. ```N``` represent the total number of grid points ans ```s``` is the elapsed time of a given kernel for a single iteration. ```f``` and ```b``` presented in Table V of the paper (section VI) are the analytical estimates from the source code.
