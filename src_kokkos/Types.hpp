#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_OffsetView.hpp>
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

template <unsigned ND> using coord_t = std::array<int, ND>;
template <unsigned ND> using shape_t = std::array<int, ND>;

typedef Kokkos::DefaultExecutionSpace execution_space;

// Views for multidimensional arrays
template <typename T> using View1D = Kokkos::View<T*, execution_space>;
template <typename T> using View2D = Kokkos::View<T**, execution_space>;
template <typename T> using View3D = Kokkos::View<T***, execution_space>;
template <typename T> using View4D = Kokkos::View<T****, execution_space>;
template <typename T> using View5D = Kokkos::View<T*****, execution_space>;

template <typename T> using OffsetView1D = Kokkos::Experimental::OffsetView<T*, execution_space>;
template <typename T> using OffsetView2D = Kokkos::Experimental::OffsetView<T**, execution_space>;
template <typename T> using OffsetView3D = Kokkos::Experimental::OffsetView<T***, execution_space>;
template <typename T> using OffsetView4D = Kokkos::Experimental::OffsetView<T****, execution_space>;
template <typename T> using OffsetView5D = Kokkos::Experimental::OffsetView<T*****, execution_space>;

typedef View1D<float64> RealView1D;
typedef View2D<float64> RealView2D;
typedef View3D<float64> RealView3D;
typedef View4D<float64> RealView4D;

typedef OffsetView1D<float64> RealOffsetView1D;
typedef OffsetView2D<float64> RealOffsetView2D;
typedef OffsetView3D<float64> RealOffsetView3D;
typedef OffsetView4D<float64> RealOffsetView4D;

typedef View1D<complex128> ComplexView1D;
typedef View2D<complex128> ComplexView2D;
typedef View3D<complex128> ComplexView3D;
typedef View4D<complex128> ComplexView4D;

typedef View1D<int> IntView1D;
typedef View2D<int> IntView2D;

// Range Policies
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_2D;
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_3D;
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<4, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_4D;
typedef typename Kokkos::MDRangePolicy< Kokkos::Rank<5, Kokkos::Iterate::Default, Kokkos::Iterate::Default> > MDPolicyType_5D;

#endif
