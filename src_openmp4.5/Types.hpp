#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <complex>
#include <omp.h>
#include "OpenMP_View.hpp"

// Directives to force vectorization
#if defined ( ENABLE_OPENMP_OFFLOAD )
  #define LOOP_SIMD
  #define SIMD_WIDTH 1
#else
  #define SIMD_WIDTH 8
  
  #if defined(SIMD)
    #if defined(FUJI)
      #define LOOP_SIMD _Pragma("loop simd")
    #else
      #define LOOP_SIMD _Pragma("omp simd")
    #endif
  #else
    #define LOOP_SIMD
  #endif
#endif

#define LONG_BUFFER_WIDTH 256

using int8  = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;

using uint8  = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using float32 = float;
using float64 = double;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
template <typename RealType> using Complex = std::complex<RealType>;

const complex128 I = complex128(0., 1.);

// RealView
using RealView1D = View<float64, 1, array_layout::value>;
using RealView2D = View<float64, 2, array_layout::value>;
using RealView3D = View<float64, 3, array_layout::value>;
using RealView4D = View<float64, 4, array_layout::value>;

using ComplexView1D = View<complex128, 1, array_layout::value>;
using ComplexView2D = View<complex128, 2, array_layout::value>;

using IntView1D = View<int, 1, array_layout::value>;
using IntView2D = View<int, 2, array_layout::value>;

#endif
