#ifndef __FFT_HPP__
#define __FFT_HPP__

#if defined( ENABLE_OPENMP_OFFLOAD )
  #include "OpenMP_Offload_FFT.hpp"
#else
  #include "OpenMP_FFT.hpp"
#endif

#endif
