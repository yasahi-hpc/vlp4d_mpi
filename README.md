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
Compilers (cuda/10.2.48, pgi19.1)

- Intel Skylake on JFRS-1 (IFERC-CSC, Japan)  
Compilers (intel19.0.0.117)

- Fujitsu A64FX on Flow (Nagoya Univ., Japan)  
Compilers (Fujitsu compiler 1.2.25)

# Usage
## Compile
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
export DEVICE=device_name # choose the device_name from "p100", "bdw", "skx", "a64fx_flow"
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


