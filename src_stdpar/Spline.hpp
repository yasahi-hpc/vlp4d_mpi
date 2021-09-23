#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include <numeric>
#include <execution>
#include <algorithm>
#include "Config.hpp"
#include "Types.hpp"
#include "Transpose.hpp"
#include "Helper.hpp"

struct Spline {
  Impl::Transpose<float64, array_layout::value> *transpose_;

private:
  RealView4D fn_trans_, fn_tmp_, fn_trans_tmp_;
  int nx_, ny_, nvx_, nvy_;
  int nx_min_, ny_min_, nvx_min_, nvy_min_;

public:
  Spline(Config *conf) {
    Domain *dom = &(conf->dom_);

    nx_min_  = dom->local_nxmin_[0] - HALO_PTS; 
    ny_min_  = dom->local_nxmin_[1] - HALO_PTS;
    nvx_min_ = dom->local_nxmin_[2] - HALO_PTS;
    nvy_min_ = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max  = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max  = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;

    nx_  = nx_max  - nx_min_;
    ny_  = ny_max  - ny_min_;
    nvx_ = nvx_max - nvx_min_;
    nvy_ = nvy_max - nvy_min_;

    // Something is wrong with Transpose kernel
    // Device Synchronization mandatory
    fn_trans_     = RealView4D("fn_trans", {nvx_,nvy_,nx_,ny_}, {nvx_min_,nvy_min_,nx_min_,ny_min_});
    fn_trans_tmp_ = RealView4D("fn_trans_tmp", {nvx_,nvy_,nx_,ny_}, {nvx_min_,nvy_min_,nx_min_,ny_min_});
    fn_tmp_       = RealView4D("fn_tmp", {nx_,ny_,nvx_,nvy_}, {nx_min_,ny_min_,nvx_min_,nvy_min_});
    fn_trans_.fill(0);
    fn_trans_tmp_.fill(0);
    fn_tmp_.fill(0);
    fn_trans_.updateDevice();
    fn_trans_tmp_.updateDevice();
    fn_tmp_.updateDevice();

    transpose_ = new Impl::Transpose<float64, array_layout::value>(nx_*ny_, nvx_*nvy_);
  };

  ~Spline() {
    if(transpose_ != nullptr) delete transpose_;
  }

  void computeCoeff_xy(RealView4D &fn, int rank) {
    computeCoeff_xy_<array_layout::value>(fn, rank);
  }
  void computeCoeff_vxvy(RealView4D &fn, int rank) {
    computeCoeff_vxvy_<array_layout::value>(fn, rank);
  }

  // Internal functions
private:
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff_xy_(RealView4D &fn, int rank) {

    #if defined( _NVHPC_STDPAR_GPU )
      transpose_->forward(fn.data(), fn_trans_.data());
      computeCoeff<array_layout::value>(fn_trans_, fn_trans_tmp_);
      transpose_->backward(fn_trans_.data(), fn.data());
    #else
      computeCoeff<array_layout::value>(fn, fn_tmp_);
    #endif
  }

  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff_xy_(RealView4D &fn, int rank) {
    #if defined( _NVHPC_STDPAR_GPU )
      computeCoeff<array_layout::value>(fn, fn_tmp_);
    #else
      transpose_->forward(fn.data(), fn_trans_.data());
      computeCoeff<array_layout::value>(fn_trans_, fn_trans_tmp_);
      transpose_->backward(fn_trans_.data(), fn.data());
    #endif
  }

  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff_vxvy_(RealView4D &fn, int rank) {
    #if defined( _NVHPC_STDPAR_GPU )
      computeCoeff<array_layout::value>(fn, fn_tmp_);
    #else
      transpose_->forward(fn.data(), fn_trans_.data());
      computeCoeff<array_layout::value>(fn_trans_, fn_trans_tmp_);
      transpose_->backward(fn_trans_.data(), fn.data());
    #endif
  }

  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff_vxvy_(RealView4D &fn, int rank) {
    #if defined( _NVHPC_STDPAR_GPU )
      transpose_->forward(fn.data(), fn_trans_.data());
      computeCoeff<array_layout::value>(fn_trans_, fn_tmp_);
      transpose_->backward(fn_trans_.data(), fn.data());
    #else
      computeCoeff<array_layout::value>(fn, fn_tmp_);
    #endif
  }

  // Layout Left
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff(RealView4D &fn, RealView4D &fn_tmp) {
    #if defined( _NVHPC_STDPAR_GPU )
      computeCoeffCore_parallel_xy(fn, fn_tmp);
    #else
      computeCoeffCore_parallel_vxvy(fn, fn_tmp);
    #endif
  }
  
  // Layout Right
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff(RealView4D &fn, RealView4D &fn_tmp) {
    #if defined( _NVHPC_STDPAR_GPU )
      computeCoeffCore_parallel_vxvy(fn, fn_tmp);
    #else
      computeCoeffCore_parallel_xy(fn, fn_tmp);
    #endif
  }

  inline void computeCoeffCore_parallel_xy(RealView4D &fn, RealView4D &fn_tmp) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);

    auto idx_4d = fn.index();
    float64 *ptr_fn = fn.data();
    float64 *ptr_fn_tmp = fn_tmp.data();

    // Define accessors by macro
    #define _fn(j0, j1, j2, j3) ptr_fn[idx_4d(j0, j1, j2, j3)]
    #define _fn_tmp(j0, j1, j2, j3) ptr_fn_tmp[idx_4d(j0, j1, j2, j3)]

    const int i2start = n2_min + HALO_PTS - 2;
    const int i2end   = n2_max - HALO_PTS + 1;
    const int i3start = n3_min + HALO_PTS - 2;
    const int i3end   = n3_max - HALO_PTS + 1;

    const int n0 = n0_max - n0_min;
    const int n1 = n1_max - n1_min;
    const int n = n0 * n1;

    Coord<2, array_layout::value> coord_2d({n0, n1}, {n0_min, n1_min});

    auto spline_2d = [=](const int idx) {
      int ptr_idx[2];
      coord_2d.to_coord(idx, ptr_idx);
      int i0 = ptr_idx[0], i1 = ptr_idx[1];
      const float64 alpha = sqrt3 - 2;
      const float64 beta  = sqrt3 * (1 - alpha * alpha);

      #if defined( LONG_ENOUGH_BUFFER )
        float64 tmp1d[LONG_WIDTH];
      #endif
      // row update
      for(int i3 = i3start-1; i3 <= i3end + 1; i3++) {

        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0, i1, i2start-1, i3) + _fn(i0, i1, i2start, i3);
          for(int nn = 1; i2start + nn <= i2end; nn++) {
            tmp1d[nn]= _fn(i0, i1, i2start+nn, i3) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0, i1, i2start, i3) = _fn(i0, i1, i2start-1, i3) + _fn(i0, i1, i2start, i3);
          for(int nn = 1; i2start + nn <= i2end; nn++) {
            _fn_tmp(i0, i1, i2start + nn, i3) = _fn(i0, i1, i2start + nn, i3) + alpha * _fn_tmp(i0, i1, i2start + nn - 1, i3);
          }
        #endif

        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0, i1, i2end+1, i3) + _fn(i0, i1, i2end, i3);
        float64 alpha_k = alpha;
        for(int nn = 1; i2start <= i2end - nn; nn++) {
          fnend += _fn(i0, i1, i2end-nn, i3) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }

        _fn(i0, i1, i2end, i3) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i2end - 1; nn >= i2start; nn--) {
            _fn(i0, i1, nn, i3) = beta * tmp1d[nn-i2start] + alpha * _fn(i0, i1, nn + 1, i3);
          }
        #else
          for(int nn = i2end - 1; nn >= i2start; nn--) {
            _fn(i0, i1, nn, i3) = beta * _fn_tmp(i0, i1, nn, i3) + alpha * _fn(i0, i1, nn + 1, i3);
          }
        #endif
      }

      // col update
      for(int i2 = i2start; i2 <= i2end; i2++) {
        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0, i1, i2, i3start-1) + _fn(i0, i1, i2, i3start);
          for(int nn = 1; i3start + nn <= i3end; nn++) {
            tmp1d[nn] = _fn(i0, i1, i2, i3start + nn) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0, i1, i2, i3start) = _fn(i0, i1, i2, i3start-1) + _fn(i0, i1, i2, i3start);
          for(int nn = 1; i3start + nn <= i3end; nn++) {
            _fn_tmp(i0, i1, i2, i3start + nn) = _fn(i0, i1, i2, i3start + nn) + alpha * _fn_tmp(i0, i1, i2, i3start + nn - 1);
          }
        #endif

        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0, i1, i2, i3end + 1) + _fn(i0, i1, i2, i3end);
        float64 alpha_k = alpha;
        for(int nn = 1; i3start <= i3end - nn; nn++) {
          fnend += _fn(i0, i1, i2, i3end - nn) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }

        _fn(i0, i1, i2, i3end) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i3end - 1; nn >= i3start; nn--) {
            _fn(i0, i1, i2, nn) = beta * tmp1d[nn-i3start] + alpha * _fn(i0, i1, i2, nn + 1);
          }
        #else
          for(int nn = i3end - 1; nn >= i3start; nn--) {
            _fn(i0, i1, i2, nn) = beta * _fn_tmp(i0, i1, i2, nn) + alpha * _fn(i0, i1, i2, nn + 1);
          }
        #endif
      }
    };

    std::for_each_n(std::execution::par_unseq,
                    counting_iterator(0), n,
                    spline_2d);
  }

  inline void computeCoeffCore_parallel_vxvy(RealView4D &fn, RealView4D &fn_tmp) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);

    auto idx_4d = fn.index();
    float64 *ptr_fn = fn.data();
    float64 *ptr_fn_tmp = fn_tmp.data();

    // Define accessors by macro
    #define _fn(j0, j1, j2, j3) ptr_fn[idx_4d(j0, j1, j2, j3)]
    #define _fn_tmp(j0, j1, j2, j3) ptr_fn_tmp[idx_4d(j0, j1, j2, j3)]

    const int i0start = n0_min + HALO_PTS - 2;
    const int i0end   = n0_max - HALO_PTS + 1;
    const int i1start = n1_min + HALO_PTS - 2;
    const int i1end   = n1_max - HALO_PTS + 1;

    const int n2 = n2_max - n2_min;
    const int n3 = n3_max - n3_min;
    const int n = n2 * n3;

    Coord<2, array_layout::value> coord_2d({n2, n3}, {n2_min, n3_min});

    auto spline_2d = [=](const int idx) {
      int ptr_idx[2];
      coord_2d.to_coord(idx, ptr_idx);
      int i2 = ptr_idx[0], i3 = ptr_idx[1];

      const float64 alpha = sqrt3 - 2;
      const float64 beta  = sqrt3 * (1 - alpha * alpha);
      #if defined( LONG_ENOUGH_BUFFER )
        float64 tmp1d[LONG_WIDTH];
      #endif
      // row update
      for(int i1 = i1start-1; i1 <= i1end + 1; i1++) {
        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0start-1, i1, i2, i3) + _fn(i0start, i1, i2, i3);
          for(int nn = 1; i0start + nn <= i0end; nn++) {
            tmp1d[nn] = _fn(i0start + nn, i1, i2, i3) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0start, i1, i2, i3) = _fn(i0start-1, i1, i2, i3) + _fn(i0start, i1, i2, i3);
          for(int nn = 1; i0start + nn <= i0end; nn++) {
            _fn_tmp(i0start+nn, i1, i2, i3) = _fn(i0start + nn, i1, i2, i3) + alpha * _fn_tmp(i0start + nn - 1, i1, i2, i3);
          }
        #endif
         
        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0end+1, i1, i2, i3) + _fn(i0end, i1, i2, i3);
        float64 alpha_k = alpha;
        for(int nn = 1; i0start <= i0end - nn; nn++) {
          fnend += _fn(i0end - nn, i1, i2, i3) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }
        
        _fn(i0end, i1, i2, i3) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i0end - 1; nn >= i0start; nn--) {
            _fn(nn, i1, i2, i3) = beta * tmp1d[nn-i0start] + alpha * _fn(nn + 1, i1, i2, i3);
          }
        #else
          for(int nn = i0end - 1; nn >= i0start; nn--) {
            _fn(nn, i1, i2, i3) = beta * _fn_tmp(nn, i1, i2, i3) + alpha * _fn(nn + 1, i1, i2, i3);
          }
        #endif
      }

      // col update
      for(int i0 = i0start; i0 <= i0end; i0++) {
        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0, i1start-1, i2, i3) + _fn(i0, i1start, i2, i3);
          for(int nn = 1; i1start + nn <= i1end; nn++) {
            tmp1d[nn] = _fn(i0, i1start + nn, i2, i3) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0, i1start, i2, i3) = _fn(i0, i1start-1, i2, i3) + _fn(i0, i1start, i2, i3);
          for(int nn = 1; i1start + nn <= i1end; nn++) {
            _fn_tmp(i0, i1start + nn, i2, i3) = _fn(i0, i1start + nn, i2, i3) + alpha * _fn_tmp(i0, i1start + nn - 1, i2, i3);
          }
        #endif
         
        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0, i1end+1, i2, i3) + _fn(i0, i1end, i2, i3);
        float64 alpha_k = alpha;
        for(int nn = 1; i1start <= i1end - nn; nn++) {
          fnend += _fn(i0, i1end - nn, i2, i3) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }
         
        _fn(i0, i1end, i2, i3) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i1end - 1; nn >= i1start; nn--) {
            _fn(i0, nn, i2, i3) = beta * tmp1d[nn-i1start] + alpha * _fn(i0, nn + 1, i2, i3);
          }
        #else
          for(int nn = i1end - 1; nn >= i1start; nn--) {
            _fn(i0, nn, i2, i3) = beta * _fn_tmp(i0, nn, i2, i3) + alpha * _fn(i0, nn + 1, i2, i3);
          }
        #endif
      }
    };
    std::for_each_n(std::execution::par_unseq,
                    counting_iterator(0), n,
                    spline_2d);
  }
};

#endif
