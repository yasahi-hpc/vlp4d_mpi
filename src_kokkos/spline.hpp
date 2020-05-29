#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include "config.h"
#include "types.h"
#include "communication.hpp"
#include "tiles.h"

#if defined( KOKKOS_ENABLE_CUDA )
  #include "Cuda_Transpose.hpp"
#else
  #include "OpenMP_Transpose.hpp"
#endif

namespace Spline {
  // prototypes
  void computeCoeff_xy(Config *conf, RealView4D fn);
  void computeCoeff_vxvy(Config *conf, RealView4D fn);

  // Internal functions
  /* 
   * @brief
   * @param[in]  tmp1d(-HALO_PTS:n0+HALO_PTS+1)
   * @param[out] fn1d(-HALO_PTS:n0+HALO_PTS+1)
   */
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  void getSplineCoeff1D(ViewType &tmp1d, ViewType &fn1d, const float64 sqrt3) {
    const int istart = HALO_PTS - 2;
    const int iend   = fn1d.extent(0) - HALO_PTS + 1;
    //const int iend   = fn1d.extent(0) + HALO_PTS + 1;
    const float64 alpha = sqrt3 - 2;
    const float64 beta  = sqrt3 * (1 - alpha * alpha);

    //// Avoid contamination of the temporarly array
    //for(int i=0; i<tmp1d.extent(0); i++) {
    //  tmp1d(i) = 0;
    //}
     
    // fn[istart-1] stores the precomputed left sum
    tmp1d(istart) = fn1d(istart-1) + fn1d(istart);
    for(int nn = 1; istart + nn <= iend; nn++) {
      tmp1d(istart + nn) = fn1d(istart + nn) + alpha * tmp1d(istart + nn - 1);
    }
     
    // fn[iend+1] stores the precomputed right sum
    float64 fnend = fn1d(iend + 1) + fn1d(iend);
    float64 alpha_k = alpha;
    for(int nn = 1; istart <= iend - nn; nn++) {
      fnend += fn1d(iend - nn) * alpha_k; //STDALGO
      alpha_k *= alpha;
    }
    
    fn1d(iend) = fnend * sqrt3;
    for(int nn = iend - 1; nn >= istart; nn--) {
      fn1d(nn) = beta * tmp1d(nn) + alpha * fn1d(nn + 1);
    }
  }

  /* 
   * @brief
   * @param[in]  tmp2d(-HALO_PTS:n0+HALO_PTS+1, -HALO_PTS:n1+HALO_PTS+1)
   * @param[out] fn2d(-HALO_PTS:n0+HALO_PTS+1, -HALO_PTS:n1+HALO_PTS+1)
   * 
   */
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION
  void getSplineCoeff2D(ViewType &tmp2d, ViewType &fn2d, float64 sqrt3, int check) {
    const int ixstart = HALO_PTS - 2;
    const int ixend   = fn2d.extent(0) - HALO_PTS + 1;
    const int iystart = HALO_PTS - 2;
    const int iyend   = fn2d.extent(1) - HALO_PTS + 1;

    // Precomputed Right and left coefficients on each column and row
    // are already start in (*, xstart-1), (*, xend+1) locations
    // All these calculations are done in halo_fill_boundary_cond:communication.cpp
    
    // Compute spline coefficients using precomputed parts
    for(int iy = iystart - 1; iy <= iyend + 1; iy++) {
      auto row     = Kokkos::subview(fn2d,  Kokkos::ALL, iy);
      auto tmp_row = Kokkos::subview(tmp2d, Kokkos::ALL, iy);
      getSplineCoeff1D(tmp_row, row, sqrt3);
      // row updated
    }

    for(int ix = ixstart; ix <= ixend; ix++) {
      auto col     = Kokkos::subview(fn2d,  ix, Kokkos::ALL);
      auto tmp_col = Kokkos::subview(tmp2d, ix, Kokkos::ALL);
      getSplineCoeff1D(tmp_col, col, sqrt3);
      // col updated
    }
  }

  struct spline_coef_2d {
    Config *conf_;
    RealView4D fn_;    // transposed to (ivx, ivy, ix, iy) for xy and (ix, iy, ivx, ivy) for vxvy
    RealView4D tmp4d_; // Buffer accessed as subviews
    float64 sqrt3_;
    int check_ = 1;

    spline_coef_2d(Config *conf, RealView4D fn)
      : conf_(conf), fn_(fn) { 
      sqrt3_ = sqrt(3); 
      int n0 = fn_.extent(0);
      int n1 = fn_.extent(1);
      int n2 = fn_.extent(2);
      int n3 = fn_.extent(3);
      tmp4d_ = RealView4D("tmp4d", n0, n1, n2, n3);
    }

    ~spline_coef_2d() {
      // Just to make sure tmp4d_ is deallocated
      Impl::free(tmp4d_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i0, const int i1) const {
      auto sub_tmp = Kokkos::subview(tmp4d_, i0, i1, Kokkos::ALL, Kokkos::ALL);
      auto sub_fn  = Kokkos::subview(fn_,    i0, i1, Kokkos::ALL, Kokkos::ALL);

      // 2D Spline interpolation
      getSplineCoeff2D(sub_tmp, sub_fn, sqrt3_, check_);
    }
  };

  void computeCoeff_xy(Config *conf, RealView4D fn) {
    int nx  = fn.extent(0);
    int ny  = fn.extent(1);
    int nvx = fn.extent(2);
    int nvy = fn.extent(3);

    Impl::Transpose<float64> transpose(nx*ny, nvx*nvy); 
    RealView4D fn_trans = RealView4D("fn_trans", nvx, nvy, nx, ny);
    transpose.forward(fn.data(), fn_trans.data());

    MDPolicyType_2D spline_xy_policy2d({{0, 0}},
                                       {{nvx, nvy}},
                                       {{TILE_SIZE0, TILE_SIZE1}}
                                      );
    Kokkos::parallel_for("spline_coef_xy", spline_xy_policy2d, spline_coef_2d(conf, fn_trans));
    transpose.backward(fn_trans.data(), fn.data());
  }

  // This is fine
  void computeCoeff_vxvy(Config *conf, RealView4D fn) {
    int nx  = fn.extent(0);
    int ny  = fn.extent(1);
    MDPolicyType_2D spline_vxvy_policy2d({{0, 0}},
                                         {{nx, ny}},
                                         {{TILE_SIZE0, TILE_SIZE1}}
                                        );
    Kokkos::parallel_for("spline_coef_vxvy", spline_vxvy_policy2d, spline_coef_2d(conf, fn));
  }
};

#endif
