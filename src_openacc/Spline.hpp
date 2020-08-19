#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include "Config.hpp"
#include "Types.hpp"

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

  // Compute spline coefficients starting at HALO_PTS-2 and ending at n0+HALO_PTS+1.
  static inline void getSplineCoeff1D(float64 fn[], const int n0, const float64 sqrt3) {
    float64 dd[n0 + 2*HALO_PTS];
    const float alpha = sqrt3 - 2;
    const int istart = HALO_PTS - 2;
    const int iend   = n0 + HALO_PTS + 1;
    const float64 beta = sqrt3 * (1 - alpha * alpha);

    // fn[istart-1] stores the precomputed left sum
    dd[istart] = (fn[istart - 1] + fn[istart]);
    for(int nn = 1; istart + nn <= iend; nn++) {
      dd[istart + nn] = fn[istart + nn] + alpha * dd[istart + nn - 1];
    }
    float64 alpha_k = alpha;

    // fn[iend+1] stores the precomputed right sum
    float64 fnend = (fn[iend + 1] + fn[iend]);
    for(int nn = 1; istart <= iend - nn; nn++) {
      fnend += fn[iend-nn] * alpha_k; //STDALGO
      alpha_k *= alpha;
    }

    fn[iend] = fnend * sqrt3;
    for(int nn = iend - 1; nn >= istart; nn--) {
      fn[nn] = beta * dd[nn] + alpha * fn[nn+1];
    }
  }

  static inline void getSplineCoeff2D(float64 fn[], const int n0, const int n1, const float64 sqrt3) {
    const int xstart = HALO_PTS - 2;
    const int xend   = n0 + HALO_PTS + 1;
    const int ystart = HALO_PTS - 2;
    const int yend   = n1 + HALO_PTS + 1;

    // Precomputed Right and left coefficients on each column and row
    // are already start in (*,xstart-1), (*,xend+1) locations
    // All these calculations are done in halo_fill_boundary_cond:communication.cpp
    
    // Compute spline coefficients using precomputed parts
    for(int y = ystart - 1; y <= yend + 1; y++) {
      float64* row = fn + y * (n0 + 2 * HALO_PTS);
      getSplineCoeff1D(row, n0, sqrt3);
    }

    for(int x = xstart; x <= xend; x++) {
      float64 col[n1 + 2 * HALO_PTS];

      for(int y = 0; y < n1 + 2 * HALO_PTS; y++)
        col[y] = fn[y * (n0 + 2 *HALO_PTS) + x];
      getSplineCoeff1D(col, n1, sqrt3);
      for(int y = 0; y < n1 + 2 * HALO_PTS; y++)
        fn[y * (n0 + 2 *HALO_PTS) + x] = col[y];
    }
  }

  // Declaration
  // Layout Left
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutLeft>>::value, void>::type
  computeCoeff_xy_(Config *conf, RealView4D &fn) {
    Domain *dom = &(conf->dom_);

    const int nx_min = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;

    //std::cout << "nx_min, ny_min, nvx_min, nvy_min = " << nx_min << ", " 
    //          << ny_min << ", " << nvx_min << ", " << nvy_min << std::endl;
    //std::cout << "nx_max, ny_max, nvx_max, nvy_max = " << nx_max << ", " 
    //          << ny_max << ", " << nvx_max << ", " << nvy_max << std::endl;

    const int nx_inner = dom->local_nx_[0];
    const int ny_inner = dom->local_nx_[1];
    const int nx = nx_inner + HALO_PTS*2;
    const int ny = ny_inner + HALO_PTS*2;
    const float64 sqrt3 = sqrt(3);
    //RealView2D tmp2d("tmp2d", nx, ny);
    //float64 *ptr_tmp2d = tmp2d.data();
    //
    //std::cout << "nx, ny = " << nx << ", " << ny << std::endl;

    // For layout Left
    //#pragma omp parallel for collapse(2) private(ptr_tmp2d)
    #pragma omp parallel for collapse(2)
    for(int ivy=nvy_min; ivy < nvy_max; ivy++) {
      for(int ivx=nvx_min; ivx < nvx_max; ivx++) {
        float64 tmp2d[nx*ny];
        for(int iy=ny_min; iy < ny_max; iy++) {
          for(int ix=nx_min; ix < nx_max; ix++) {
            int idx = Index::coord_2D2int(ix-nx_min, iy-ny_min, nx, ny);
            tmp2d[idx] = fn(ix, iy, ivx, ivy);
            //ptr_tmp2d[idx] = fn(ix, iy, ivx, ivy);
          }
        }
        getSplineCoeff2D(tmp2d, nx_inner, ny_inner, sqrt3);
        for(int iy=ny_min; iy < ny_max; iy++) {
          for(int ix=nx_min; ix < nx_max; ix++) {
            int idx = Index::coord_2D2int(ix-nx_min, iy-ny_min, nx, ny);
            fn(ix, iy, ivx, ivy) = tmp2d[idx];
          }
        }
      }
    }
  }

  // Layout right
  template <Layout LayoutType>
    typename std::enable_if<std::is_same<std::integral_constant<Layout, LayoutType>,
                                         std::integral_constant<Layout, Layout::LayoutRight>>::value, void>::type
  computeCoeff_xy_(Config *conf, RealView4D &fn) {
    Domain *dom = &(conf->dom_);

    const int nx_min = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;

    const int nx = dom->local_nx_[0];
    const int ny = dom->local_nx_[1];
    const float64 sqrt3 = sqrt(3);
    RealView2D tmp2d("tmp2d", nx, ny);
    float64 *ptr_tmp2d = tmp2d.data();

    #pragma omp parallel for collapse(2) private(ptr_tmp2d)
    for(int ivy=nvy_min; ivy < nvy_max; ivy++) {
      for(int ivx=nvx_min; ivx < nvx_max; ivx++) {
        for(int iy=ny_min; iy < ny_max; iy++) {
          for(int ix=nx_min; ix < nx_max; ix++) {
            int idx = Index::coord_2D2int(ix-nx_min, iy-ny_min, nx, ny);
            ptr_tmp2d[idx] = fn(ix, iy, ivx, ivy);
          }
        }
        getSplineCoeff2D(ptr_tmp2d, nx, ny, sqrt3);
        for(int iy=ny_min; iy < ny_max; iy++) {
          for(int ix=nx_min; ix < nx_max; ix++) {
            int idx = Index::coord_2D2int(ix-nx_min, iy-ny_min, nx, ny);
            fn(ix, iy, ivx, ivy) = ptr_tmp2d[idx];
          }
        }
      }
    }
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

    const float64 sqrt3 = sqrt(3);
    const int nvx_inner = dom->local_nx_[2];
    const int nvy_inner = dom->local_nx_[3];
    const int nvx = nvx_inner + HALO_PTS*2;
    const int nvy = nvy_inner + HALO_PTS*2;
    //RealView2D tmp2d("tmp2d", nvx, nvy);
    //float64 *ptr_tmp2d = tmp2d.data();
    //
    std::cout << "nx_min, ny_min, nvx_min, nvy_min = " << nx_min << ", " 
              << ny_min << ", " << nvx_min << ", " << nvy_min << std::endl;
    std::cout << "nx_max, ny_max, nvx_max, nvy_max = " << nx_max << ", " 
              << ny_max << ", " << nvx_max << ", " << nvy_max << std::endl;
    std::cout << "nvx_inner, nvy_inner = " << nvx_inner << ", " 
              << nvy_inner << std::endl;
    

    // For layout Left
    //#pragma omp parallel for collapse(2) private(ptr_tmp2d)
    #pragma omp parallel for collapse(2)
    for(int iy=ny_min; iy < ny_max; iy++) {
      for(int ix=nx_min; ix < nx_max; ix++) {
        float64 tmp2d[nvx*nvy];
        for(int ivy=nvy_min; ivy < nvy_max; ivy++) {
          for(int ivx=nvx_min; ivx < nvx_max; ivx++) {
            int idx = Index::coord_2D2int(ivx-nvx_min, ivy-nvy_min, nvx, nvy);
            tmp2d[idx] = fn(ix, iy, ivx, ivy);
          }
        }
        getSplineCoeff2D(tmp2d, nvx_inner, nvy_inner, sqrt3);
        for(int ivy=nvy_min; ivy < nvy_max; ivy++) {
          for(int ivx=nvx_min; ivx < nvx_max; ivx++) {
            int idx = Index::coord_2D2int(ivx-nvx_min, ivy-nvy_min, nvx, nvy);
            fn(ix, iy, ivx, ivy) = tmp2d[idx];
          }
        }
      }
    }
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

    const int nvx = dom->local_nx_[2];
    const int nvy = dom->local_nx_[3];
    const float64 sqrt3 = sqrt(3);
    RealView2D tmp2d("tmp2d", nvx, nvy);
    float64 *ptr_tmp2d = tmp2d.data();

    // For layout Right
    #pragma omp parallel for collapse(2) private(ptr_tmp2d)
    for(int iy=ny_min; iy < ny_max; iy++) {
      for(int ix=nx_min; ix < nx_max; ix++) {
        for(int ivy=nvy_min; ivy < nvy_max; ivy++) {
          for(int ivx=nvx_min; ivx < nvx_max; ivx++) {
            int idx = Index::coord_2D2int(ivx-nvx_min, ivy-nvy_min, nvx, nvy);
            ptr_tmp2d[idx] = fn(ix, iy, ivx, ivy);
          }
        }
        getSplineCoeff2D(ptr_tmp2d, nvx, nvy, sqrt3);
        for(int ivy=nvy_min; ivy < nvy_max; ivy++) {
          for(int ivx=nvx_min; ivx < nvx_max; ivx++) {
            int idx = Index::coord_2D2int(ivx-nvx_min, ivy-nvy_min, nvx, nvy);
            fn(ix, iy, ivx, ivy) = ptr_tmp2d[idx];
          }
        }
      }
    }
  }
};

#endif
