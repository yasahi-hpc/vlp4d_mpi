#ifndef __TYPES_H__
#define __TYPES_H__

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <array>

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
typedef Kokkos::complex<float32> complex64;
typedef Kokkos::complex<float64> complex128;
// const complex64 I = complex64(0., 1.); // for some reason, it does not work for GPU version

template <unsigned ND> using coord_t = std::array<int, ND>;
template <unsigned ND> using shape_t = std::array<int, ND>;

typedef Kokkos::DefaultExecutionSpace execution_space;

// Views for multidimensional arrays
template <typename T> using View1D = Kokkos::View<T*, execution_space>;
template <typename T> using View2D = Kokkos::View<T**, execution_space>;
template <typename T> using View3D = Kokkos::View<T***, execution_space>;
template <typename T> using View4D = Kokkos::View<T****, execution_space>;
template <typename T> using View5D = Kokkos::View<T*****, execution_space>;

template <typename T> using LeftView1D = Kokkos::View<T*, Kokkos::LayoutLeft, execution_space>;
template <typename T> using LeftView2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
template <typename T> using LeftView3D = Kokkos::View<T***, Kokkos::LayoutLeft, execution_space>;
template <typename T> using LeftView4D = Kokkos::View<T****, Kokkos::LayoutLeft, execution_space>;
template <typename T> using LeftView5D = Kokkos::View<T*****, Kokkos::LayoutLeft, execution_space>;

typedef View1D<float64> RealView1D;
typedef View2D<float64> RealView2D;
typedef View3D<float64> RealView3D;
typedef View4D<float64> RealView4D;
typedef RealView4D::HostMirror RealView4Dhost;

typedef LeftView1D<float64> RealLeftView1D;
typedef LeftView2D<float64> RealLeftView2D;
typedef LeftView3D<float64> RealLeftView3D;
typedef LeftView4D<float64> RealLeftView4D;

typedef View1D<complex128> ComplexView1D;
typedef View2D<complex128> ComplexView2D;
typedef View3D<complex128> ComplexView3D;
typedef View4D<complex128> ComplexView4D;


// Range Policies
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_2D;
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_3D;
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<4, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_4D;
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<5, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_5D;

struct double_pair {
  double x, y;
  KOKKOS_INLINE_FUNCTION
  double_pair(double xinit, double yinit) 
    : x(xinit), y(yinit) {};

  KOKKOS_INLINE_FUNCTION
  double_pair()
    : x(0.), y(0.) {};

  KOKKOS_INLINE_FUNCTION
  double_pair& operator += (const double_pair& src) {
    x += src.x;
    y += src.y;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile double_pair& operator += (const volatile double_pair& src) volatile {
    x += src.x;
    y += src.y;
    return *this;
  }
};

struct double_3 {
  double x, y, z;

  KOKKOS_INLINE_FUNCTION
  double_3(double xinit, double yinit, double zinit)
    : x(xinit), y(yinit), z(zinit) {}

  KOKKOS_INLINE_FUNCTION
  double_3()
    : x(0), y(0), z(0) {}

  KOKKOS_INLINE_FUNCTION
  double_3& operator += (const double_3& src) {
    x += src.x;
    y += src.y;
    z += src.z;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile double_3& operator += (const volatile double_3& src) volatile {
    x += src.x;
    y += src.y;
    z += src.z;
    return *this;
  }
};

template<typename ScalarType>
struct Scalar2 {
  ScalarType x, y;

  KOKKOS_INLINE_FUNCTION
  Scalar2(ScalarType xinit, ScalarType yinit)
    : x(xinit), y(yinit) {}

  KOKKOS_INLINE_FUNCTION
  Scalar2()
    : x(0), y(0) {}

  KOKKOS_INLINE_FUNCTION
  Scalar2& operator += (const Scalar2& src) {
    x += src.x;
    y += src.y;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile Scalar2& operator += (const volatile Scalar2& src) volatile {
    x += src.x;
    y += src.y;
    return *this;
  }
};

template<typename ScalarType>
struct Scalar3 {
  ScalarType x, y, z;

  KOKKOS_INLINE_FUNCTION
  Scalar3(ScalarType xinit, ScalarType yinit, ScalarType zinit)
    : x(xinit), y(yinit), z(zinit) {}

  KOKKOS_INLINE_FUNCTION
  Scalar3()
    : x(0), y(0), z(0) {}

  KOKKOS_INLINE_FUNCTION
  Scalar3& operator += (const Scalar3& src) {
    x += src.x;
    y += src.y;
    z += src.z;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile Scalar3& operator += (const volatile Scalar3& src) volatile {
    x += src.x;
    y += src.y;
    z += src.z;
    return *this;
  }
};

template<typename ScalarType>
struct Scalar4 {
  ScalarType x, y, z, w;

  KOKKOS_INLINE_FUNCTION
  Scalar4(ScalarType xinit, ScalarType yinit, ScalarType zinit, ScalarType winit)
    : x(xinit), y(yinit), z(zinit), w(winit) {}

  KOKKOS_INLINE_FUNCTION
  Scalar4()
    : x(0), y(0), z(0), w(0) {}

  KOKKOS_INLINE_FUNCTION
  Scalar4& operator += (const Scalar4& src) {
    x += src.x;
    y += src.y;
    z += src.z;
    w += src.w;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile Scalar4& operator += (const volatile Scalar4& src) volatile {
    x += src.x;
    y += src.y;
    z += src.z;
    w += src.w;
    return *this;
  }
};

#if ! defined( KOKKOS_ENABLE_CUDA )
struct int2 {
  int x, y;
};

KOKKOS_INLINE_FUNCTION
int2 make_int2(int x, int y) {
  int2 t; t.x = x; t.y= y; return t;
};

struct int3 {
  int x, y, z;
};

KOKKOS_INLINE_FUNCTION
int3 make_int3(int x, int y, int z) {
  int3 t; t.x = x; t.y = y; t.z = z; return t;
};

struct int4 {
  int x, y, z, w;
};

KOKKOS_INLINE_FUNCTION
int4 make_int4(int x, int y, int z, int w) {
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
};

#endif

#endif
