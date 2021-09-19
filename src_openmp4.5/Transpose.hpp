#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#if defined( ENABLE_OPENMP_OFFLOAD )
  #include "OpenMP_Offload_Transpose.hpp"
#else
  #include "OpenMP_Transpose.hpp"
#endif

#endif
