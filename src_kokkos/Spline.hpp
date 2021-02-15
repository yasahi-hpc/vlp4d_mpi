#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include "Config.hpp"
#include "Types.hpp"
#include "Communication.hpp"
#include "tiles.h"
#include "Transpose.hpp"
#include "Math.hpp"

struct Spline {
  typedef typename RealOffsetView4D::array_layout array_layout;
  Impl::Transpose<float64, array_layout> *transpose_;

private:
  RealOffsetView4D fn_trans_;
  int nx_, ny_, nvx_, nvy_;

public:
  Spline(Config *conf) {
    Domain* dom = &conf->dom_;
    nx_  = dom->local_nx_[0] + HALO_PTS * 2;
    ny_  = dom->local_nx_[1] + HALO_PTS * 2;
    nvx_ = dom->local_nx_[2] + HALO_PTS * 2;
    nvy_ = dom->local_nx_[3] + HALO_PTS * 2;

    transpose_ = new Impl::Transpose<float64, array_layout>(nx_*ny_, nvx_*nvy_);
    int nx_min = dom->local_nxmin_[0], ny_min = dom->local_nxmin_[1], nvx_min = dom->local_nxmin_[2], nvy_min = dom->local_nxmin_[3];
    int nx_max = dom->local_nxmax_[0], ny_max = dom->local_nxmax_[1], nvx_max = dom->local_nxmax_[2], nvy_max = dom->local_nxmax_[3];
    fn_trans_ = RealOffsetView4D("fn_trans",
                                 {nvx_min-HALO_PTS, nvx_max+HALO_PTS},
                                 {nvy_min-HALO_PTS, nvy_max+HALO_PTS},
                                 {nx_min-HALO_PTS, nx_max+HALO_PTS},
                                 {ny_min-HALO_PTS, ny_max+HALO_PTS}
                                );
  }

  ~Spline() {
    if(transpose_ != nullptr) delete transpose_;
  };

  void computeCoeff_xy(Config *conf, RealOffsetView4D fn, 
                       const std::vector<int> &tiles={TILE_SIZE0, TILE_SIZE1});
  void computeCoeff_vxvy(Config *conf, RealOffsetView4D fn, 
                         const std::vector<int> &tiles={TILE_SIZE0, TILE_SIZE1});

public:
  // Kokkos functions and functors
  struct spline_coef_2d {
    Config *conf_;
    RealOffsetView4D fn_;    // transposed to (ivx, ivy, ix, iy) for xy and (ix, iy, ivx, ivy) for vxvy
    RealOffsetView4D tmp4d_; // Buffer accessed as subviews
    float64 sqrt3_;
    int check_ = 1;
    int n0_min_;
    int n1_min_;
    int n2_min_;
    int n3_min_;
  
    spline_coef_2d(Config *conf, RealOffsetView4D fn)
      : conf_(conf), fn_(fn) { 
      sqrt3_ = sqrt(3); 
      int n0_min = fn_.begin(0), n1_min = fn_.begin(1), n2_min = fn_.begin(2), n3_min = fn_.begin(3);
      int n0_max = fn_.end(0)-1, n1_max = fn_.end(1)-1, n2_max = fn_.end(2)-1, n3_max = fn_.end(3) - 1;
      tmp4d_ = RealOffsetView4D("tmp4d", 
                                {n0_min, n0_max}, 
                                {n1_min, n1_max}, 
                                {n2_min, n2_max}, 
                                {n3_min, n3_max});
      n0_min_ = fn_.begin(0);
      n1_min_ = fn_.begin(1);
      n2_min_ = fn_.begin(2);
      n3_min_ = fn_.begin(3);
    }
  
    ~spline_coef_2d() {
      // Just to make sure tmp4d_ is deallocated
      Impl::free(tmp4d_);
    }

    // Internal functions
    /* 
     * @brief
     * @param[in]  tmp1d(-HALO_PTS:n0+HALO_PTS+1)
     * @param[out] fn1d(-HALO_PTS:n0+HALO_PTS+1)
     */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    void getSplineCoeff1D(ViewType &tmp1d, ViewType &fn1d, const float64 sqrt3) const {
      const int istart = fn1d.begin(0) + HALO_PTS - 2;
      const int iend   = fn1d.end(0)   - HALO_PTS + 1;
      const float64 alpha = sqrt3 - 2;
      const float64 beta  = sqrt3 * (1 - alpha * alpha);
    
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
    void getSplineCoeff2D(ViewType &tmp2d, ViewType &fn2d, float64 sqrt3, int check) const {
      const int ixstart = fn2d.begin(0) + HALO_PTS - 2;
      const int ixend   = fn2d.end(0)   - HALO_PTS + 1;
      const int iystart = fn2d.begin(1) + HALO_PTS - 2;
      const int iyend   = fn2d.end(1)   - HALO_PTS + 1;
    
      // Precomputed Right and left coefficients on each column and row
      // are already start in (*, xstart-1), (*, xend+1) locations
      // All these calculations are done in halo_fill_boundary_cond:communication.cpp
      
      // Compute spline coefficients using precomputed parts
      for(int iy = iystart - 1; iy <= iyend + 1; iy++) {
        auto row     = Kokkos::Experimental::subview(fn2d,  Kokkos::ALL, iy);
        auto tmp_row = Kokkos::Experimental::subview(tmp2d, Kokkos::ALL, iy);
        getSplineCoeff1D(tmp_row, row, sqrt3);
        // row updated
      }
    
      for(int ix = ixstart; ix <= ixend; ix++) {
        auto col     = Kokkos::Experimental::subview(fn2d,  ix, Kokkos::ALL);
        auto tmp_col = Kokkos::Experimental::subview(tmp2d, ix, Kokkos::ALL);
        getSplineCoeff1D(tmp_col, col, sqrt3);
        // col updated
      }
    }
  
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i0, const int i1) const {
      #if defined ( LAYOUT_LEFT )
        // For LayoutLeft specialization for CPU
        auto sub_tmp = Kokkos::Experimental::subview(tmp4d_, Kokkos::ALL, Kokkos::ALL, i2+n2_min_, i3+n3_min_);
        auto sub_fn  = Kokkos::Experimental::subview(fn_,    Kokkos::ALL, Kokkos::ALL, i2+n2_min_, i3+n3_min_);
      #else
        auto sub_tmp = Kokkos::Experimental::subview(tmp4d_, i0+n0_min_, i1+n1_min_, Kokkos::ALL, Kokkos::ALL);
        auto sub_fn  = Kokkos::Experimental::subview(fn_,    i0+n0_min_, i1+n1_min_, Kokkos::ALL, Kokkos::ALL);
      #endif
  
      // 2D Spline interpolation
      getSplineCoeff2D(sub_tmp, sub_fn, sqrt3_, check_);
    }
  };

}; 

#endif
