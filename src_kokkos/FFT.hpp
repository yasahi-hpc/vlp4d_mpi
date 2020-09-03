#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined( KOKKOS_ENABLE_CUDA )
  #include "Cuda_FFT.hpp"
#else
  #include "OpenMP_FFT.hpp"
#endif

#endif
