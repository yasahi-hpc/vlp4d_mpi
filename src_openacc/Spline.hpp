#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include "Config.hpp"
#include "Types.hpp"
#include "Transpose.hpp"

namespace Spline {
  // prototypes
  void computeCoeff_xy(Config *conf, RealView4D &fn);
  void computeCoeff_vxvy(Config *conf, RealView4D &fn);

  // Internal functions
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff_xy_(Config *conf, RealView4D &fn);
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff_xy_(Config *conf, RealView4D &fn);
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff_vxvy_(Config *conf, RealView4D &fn);
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff_vxvy_(Config *conf, RealView4D &fn);

  // Declaration
  void computeCoeff_xy(Config *conf, RealView4D &fn) {
    typedef typename RealView4D::layout_ array_layout;
    computeCoeff_xy_<array_layout::value>(conf, fn);
  }

  void computeCoeff_vxvy(Config *conf, RealView4D &fn) {
    typedef typename RealView4D::layout_ array_layout;
    computeCoeff_vxvy_<array_layout::value>(conf, fn);
  }

  // Common interface for OpenACC/OpenMP, fn_tmp is used as a buffer
  // Layout Left
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff(RealView4D &fn, RealView4D &fn_tmp) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);

    #if defined( ENABLE_OPENACC )
      const int i2start = n2_min + HALO_PTS - 2;
      const int i2end   = n2_max - HALO_PTS + 1;
      const int i3start = n3_min + HALO_PTS - 2;
      const int i3end   = n3_max - HALO_PTS + 1;
      #pragma acc data present(fn, fn_tmp)
      #pragma acc parallel loop collapse(2)
      for(int i1=n1_min; i1 < n1_max; i1++) {
        for(int i0=n0_min; i0 < n0_max; i0++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          // row update
          for(int i3 = i3start-1; i3 <= i3end + 1; i3++) {

            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0, i1, i2start, i3) = (fn(i0, i1, i2start-1, i3) + fn(i0, i1, i2start, i3));
            for(int nn = 1; i2start + nn <= i2end; nn++) {
              fn_tmp(i0, i1, i2start + nn, i3) = fn(i0, i1, i2start + nn, i3) + alpha * fn_tmp(i0, i1, i2start + nn - 1, i3);
            }

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2end + 1, i3) + fn(i0, i1, i2end, i3)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i2start <= i2end - nn; nn++) {
              fnend += fn(i0, i1, i2end - nn, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            fn(i0, i1, i2end, i3) = fnend * sqrt3;
            for(int nn = i2end - 1; nn >= i2start; nn--) {
              fn(i0, i1, nn, i3) = beta * fn_tmp(i0, i1, nn, i3) + alpha * fn(i0, i1, nn + 1, i3);
            }
          }

          // col update
          for(int i2 = i2start; i2 <= i2end; i2++) {
            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0, i1, i2, i3start) = (fn(i0, i1, i2, i3start-1) + fn(i0, i1, i2, i3start));
            for(int nn = 1; i3start + nn <= i3end; nn++) {
              fn_tmp(i0, i1, i2, i3start + nn) = fn(i0, i1, i2, i3start + nn) + alpha * fn_tmp(i0, i1, i2, i3start + nn - 1);
            }

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2, i3end + 1) + fn(i0, i1, i2, i3end)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i3start <= i3end - nn; nn++) {
              fnend += fn(i0, i1, i2, i3end - nn) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            fn(i0, i1, i2, i3end) = fnend * sqrt3;
            for(int nn = i3end - 1; nn >= i3start; nn--) {
              fn(i0, i1, i2, nn) = beta * fn_tmp(i0, i1, i2, nn) + alpha * fn(i0, i1, i2, nn + 1);
            }
          }
        }
      }
    #else
      const int i0start = n0_min + HALO_PTS - 2;
      const int i0end   = n0_max - HALO_PTS + 1;
      const int i1start = n1_min + HALO_PTS - 2;
      const int i1end   = n1_max - HALO_PTS + 1;
      #pragma omp parallel for collapse(2)
      for(int i3=n3_min; i3 < n3_max; i3++) {
        for(int i2=n2_min; i2 < n2_max; i2++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          // row update
          for(int i1 = i1start-1; i1 <= i1end + 1; i1++) {
            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0start, i1, i2, i3) = (fn(i0start-1, i1, i2, i3) + fn(i0start, i1, i2, i3));
            for(int nn = 1; i0start + nn <= i0end; nn++) {
              fn_tmp(i0start+nn, i1, i2, i3) = fn(i0start + nn, i1, i2, i3) + alpha * fn_tmp(i0start + nn - 1, i1, i2, i3);
            }
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0end+1, i1, i2, i3) + fn(i0end, i1, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i0start <= i0end - nn; nn++) {
              fnend += fn(i0end - nn, i1, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
            
            fn(i0end, i1, i2, i3) = fnend * sqrt3;
            for(int nn = i0end - 1; nn >= i0start; nn--) {
              fn(nn, i1, i2, i3) = beta * fn_tmp(nn, i1, i2, i3) + alpha * fn(nn + 1, i1, i2, i3);
            }
          }

          // col update
          for(int i0 = i0start; i0 <= i0end; i0++) {
            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0, i1start, i2, i3) = (fn(i0, i1start-1, i2, i3) + fn(i0, i1start, i2, i3));
            for(int nn = 1; i1start + nn <= i1end; nn++) {
              fn_tmp(i0, i1start + nn, i2, i3) = fn(i0, i1start + nn, i2, i3) + alpha * fn_tmp(i0, i1start + nn - 1, i2, i3);
            }
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1end+1, i2, i3) + fn(i0, i1end, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i1start <= i1end - nn; nn++) {
              fnend += fn(i0, i1end - nn, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
             
            fn(i0, i1end, i2, i3) = fnend * sqrt3;
            for(int nn = i1end - 1; nn >= i1start; nn--) {
              fn(i0, nn, i2, i3) = beta * fn_tmp(i0, nn, i2, i3) + alpha * fn(i0, nn + 1, i2, i3);
            }
          }
        }
      }
    #endif
  }

  // Layout Right
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff(RealView4D &fn) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);
    #if defined( ENABLE_OPENACC )
      const int i0start = n0_min + HALO_PTS - 2;
      const int i0end   = n0_max - HALO_PTS + 1;
      const int i1start = n1_min + HALO_PTS - 2;
      const int i1end   = n1_max - HALO_PTS + 1;
      #pragma acc data present(fn, fn_tmp)
      #pragma acc parallel loop collapse(2)
      for(int i2=n2_min; i2 < n2_max; i2++) {
        for(int i3=n3_min; i3 < n3_max; i3++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          // row update
          for(int i1 = i1start-1; i1 <= i1end + 1; i1++) {
            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0start, i1, i2, i3) = (fn(i0start-1, i1, i2, i3) + fn(i0start, i1, i2, i3));
            for(int nn = 1; i0start + nn <= i0end; nn++) {
              fn_tmp(i0start+nn, i1, i2, i3) = fn(i0start + nn, i1, i2, i3) + alpha * fn_tmp(i0start + nn - 1, i1, i2, i3);
            }
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0end+1, i1, i2, i3) + fn(i0end, i1, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i0start <= i0end - nn; nn++) {
              fnend += fn(i0end - nn, i1, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
            
            fn(i0end, i1, i2, i3) = fnend * sqrt3;
            for(int nn = i0end - 1; nn >= i0start; nn--) {
              fn(nn, i1, i2, i3) = beta * fn_tmp(nn, i1, i2, i3) + alpha * fn(nn + 1, i1, i2, i3);
            }
          }

          // col update
          for(int i0 = i0start; i0 <= i0end; i0++) {
            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0, i1start, i2, i3) = (fn(i0, i1start-1, i2, i3) + fn(i0, i1start, i2, i3));
            for(int nn = 1; i1start + nn <= i1end; nn++) {
              fn_tmp(i0, i1start + nn, i2, i3) = fn(i0, i1start + nn, i2, i3) + alpha * fn_tmp(i0, i1start + nn - 1, i2, i3);
            }
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1end+1, i2, i3) + fn(i0, i1end, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i1start <= i1end - nn; nn++) {
              fnend += fn(i0, i1end - nn, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
             
            fn(i0, i1end, i2, i3) = fnend * sqrt3;
            for(int nn = i1end - 1; nn >= i1start; nn--) {
              fn(i0, nn, i2, i3) = beta * fn_tmp(i0, nn, i2, i3) + alpha * fn(i0, nn + 1, i2, i3);
            }
          }
        }
      }
    #else
      const int i2start = n2_min + HALO_PTS - 2;
      const int i2end   = n2_max - HALO_PTS + 1;
      const int i3start = n3_min + HALO_PTS - 2;
      const int i3end   = n3_max - HALO_PTS + 1;
      #pragma omp parallel for collapse(2)
      for(int i1=n1_min; i1 < n1_max; i1++) {
        for(int i0=n0_min; i0 < n0_max; i0++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          // row update
          for(int i3 = i3start-1; i3 <= i3end + 1; i3++) {

            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0, i1, i2start, i3) = (fn(i0, i1, i2start-1, i3) + fn(i0, i1, i2start, i3));
            for(int nn = 1; i2start + nn <= i2end; nn++) {
              fn_tmp(i0, i1, i2start + nn, i3) = fn(i0, i1, i2start + nn, i3) + alpha * fn_tmp(i0, i1, i2start + nn - 1, i3);
            }

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2end + 1, i3) + fn(i0, i1, i2end, i3)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i2start <= i2end - nn; nn++) {
              fnend += fn(i0, i1, i2end - nn, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            fn(i0, i1, i2end, i3) = fnend * sqrt3;
            for(int nn = i2end - 1; nn >= i2start; nn--) {
              fn(i0, i1, nn, i3) = beta * fn_tmp(i0, i1, nn, i3) + alpha * fn(i0, i1, nn + 1, i3);
            }
          }

          // col update
          for(int i2 = i2start; i2 <= i2end; i2++) {
            // fn[istart-1] stores the precomputed left sum
            fn_tmp(i0, i1, i2, i3start) = (fn(i0, i1, i2, i3start-1) + fn(i0, i1, i2, i3start));
            for(int nn = 1; i3start + nn <= i3end; nn++) {
              fn_tmp(i0, i1, i2, i3start + nn) = fn(i0, i1, i2, i3start + nn) + alpha * fn_tmp(i0, i1, i2, i3start + nn - 1);
            }

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2, i3end + 1) + fn(i0, i1, i2, i3end)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i3start <= i3end - nn; nn++) {
              fnend += fn(i0, i1, i2, i3end - nn) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            fn(i0, i1, i2, i3end) = fnend * sqrt3;
            for(int nn = i3end - 1; nn >= i3start; nn--) {
              fn(i0, i1, i2, nn) = beta * fn_tmp(i0, i1, i2, nn) + alpha * fn(i0, i1, i2, nn + 1);
            }
          }
        }
      }
    #endif
  }

  // Declaration
  // Layout Left
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff_xy_(Config *conf, RealView4D &fn) {
    Domain *dom = &(conf->dom_);

    const int nx_min  = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min  = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max  = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max  = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;

    const int nx  = nx_max - nx_min;
    const int ny  = ny_max - ny_min;
    const int nvx = nvx_max - nvx_min;
    const int nvy = nvy_max - nvy_min;

    #if defined( ENABLE_OPENACC )
      RealView4D fn_trans = RealView4D("fn_trans", {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      RealView4D fn_tmp   = RealView4D("fn_tmp",   {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      Impl::Transpose<float64> transpose(nx*ny, nvx*nvy);
      transpose.forward(fn.data(), fn_trans.data());
      computeCoeff<array_layout::value>(fn_trans, fn_tmp);
      transpose.backward(fn_trans.data(), fn.data());
    #else
      RealView4D fn_tmp = RealView4D("fn_tmp", {nx,ny,nvx,nvy}, {nx_min,ny_min,nvx_min,nvy_min});
      computeCoeff<array_layout::value>(fn, fn_tmp);
    #endif
  }

  // Layout right
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff_xy_(Config *conf, RealView4D &fn) {
    Domain *dom = &(conf->dom_);

    const int nx_min  = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min  = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max  = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max  = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;

    const int nx  = nx_max - nx_min;
    const int ny  = ny_max - ny_min;
    const int nvx = nvx_max - nvx_min;
    const int nvy = nvy_max - nvy_min;

    #if defined( ENABLE_OPENACC )
      RealView4D fn_tmp = RealView4D("fn_tmp", {nx,ny,nvx,nvy}, {nx_min,ny_min,nvx_min,nvy_min});
      computeCoeff<array_layout::value>(fn, fn_tmp);
    #else
      RealView4D fn_trans = RealView4D("fn_trans", {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      RealView4D fn_tmp   = RealView4D("fn_tmp",   {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      Impl::Transpose<float64> transpose(nx*ny, nvx*nvy);
      transpose.forward(fn.data(), fn_trans.data());
      computeCoeff<array_layout::value>(fn_trans, fn_tmp);
      transpose.backward(fn_trans.data(), fn.data());
    #endif
  }

  // Layout Left
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff_vxvy_(Config *conf, RealView4D &fn) {
    Domain *dom = &(conf->dom_);

    const int nx_min = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;
    const int nx  = nx_max - nx_min;
    const int ny  = ny_max - ny_min;
    const int nvx = nvx_max - nvx_min;
    const int nvy = nvy_max - nvy_min;

    #if defined( ENABLE_OPENACC )
      RealView4D fn_tmp = RealView4D("fn_tmp", {nx,ny,nvx,nvy}, {nx_min,ny_min,nvx_min,nvy_min});
      computeCoeff<array_layout::value>(fn, fn_tmp);
    #else
      RealView4D fn_trans = RealView4D("fn_trans", {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      RealView4D fn_tmp   = RealView4D("fn_tmp",   {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      Impl::Transpose<float64> transpose(nx*ny, nvx*nvy);
      transpose.forward(fn.data(), fn_trans.data());
      computeCoeff<array_layout::value>(fn_trans, fn_tmp);
      transpose.backward(fn_trans.data(), fn.data());
    #endif
  }

  // Layout Right
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff_vxvy_(Config *conf, RealView4D &fn) {
    Domain *dom = &(conf->dom_);

    const int nx_min = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;
    const int nx  = nx_max - nx_min;
    const int ny  = ny_max - ny_min;
    const int nvx = nvx_max - nvx_min;
    const int nvy = nvy_max - nvy_min;

    #if defined( ENABLE_OPENACC )
      RealView4D fn_trans = RealView4D("fn_trans", {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      RealView4D fn_tmp   = RealView4D("fn_tmp",   {nvx,nvy,nx,ny}, {nvx_min,nvy_min,nx_min,ny_min});
      Impl::Transpose<float64> transpose(nx*ny, nvx*nvy);
      transpose.forward(fn.data(), fn_trans.data());
      computeCoeff<array_layout::value>(fn_trans, fn_tmp);
      transpose.backward(fn_trans.data(), fn.data());
    #else
      RealView4D fn_tmp = RealView4D("fn_tmp",  {nx,ny,nvx,nvy}, {nx_min,ny_min,nvx_min,nvy_min});
      computeCoeff<array_layout::value>(fn, fn_tmp);
    #endif
  }
};

#endif
