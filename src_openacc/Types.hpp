#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <complex>
#include "OpenACC_View.hpp"

// Directive to force vectorization
#if defined( ENABLE_OPENACC )
  #include <openacc.h>
  #define LOOP_SIMD _Pragma("acc loop vector independent")
#else
  #include <omp.h>
  
  #if defined(SIMD)
    #define LOOP_SIMD _Pragma("omp simd")
  #else
    #define LOOP_SIMD
  #endif
#endif

typedef int8_t  int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
 
typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef float  float32;
typedef double float64;
typedef std::complex<float>  complex64;
typedef std::complex<double> complex128;
template <typename RealType> using Complex = std::complex<RealType>;

const complex128 I = complex128(0., 1.);

// RealView 
typedef View<float64, 1, array_layout::value> RealView1D;
typedef View<float64, 2, array_layout::value> RealView2D;
typedef View<float64, 3, array_layout::value> RealView3D;
typedef View<float64, 4, array_layout::value> RealView4D;

typedef View<complex128, 1, array_layout::value> ComplexView1D;
typedef View<complex128, 2, array_layout::value> ComplexView2D;
typedef View<complex128, 3, array_layout::value> ComplexView3D;
typedef View<complex128, 4, array_layout::value> ComplexView4D;

typedef View<int, 1, array_layout::value> IntView1D;
typedef View<int, 2, array_layout::value> IntView2D;

#endif
