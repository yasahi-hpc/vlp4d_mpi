#ifndef __ADVECTION_HPP__
#define __ADVECTION_HPP__

#include "config.h"
#include "types.h"
#include "efield.hpp"
#include "Math.hpp"
#include <cmath>
#include <Kokkos_ScatterView.hpp>
#include "communication.hpp"

// Temporarly solution
#if ! defined( KOKKOS_ENABLE_CUDA )
  using std::max; using std::min;
#endif

namespace Advection {
  const int granularity[4] = {2, 2, 2, 2};

  void advect_2D_xy(Config *conf, RealView4D fn, float64 dt);
  void advect_4D(Config *conf, Efield *ef, RealView4D fn, RealView4D tmp_fn, float64 dt);
  void print_fxvx(Config *conf, Distrib &comm, const RealView4D fn, int iter);

  static void testError(View1D<int> &err) {
    typename View1D<int>::HostMirror h_err = Kokkos::create_mirror_view(err);
    if(h_err(0) != 0) {
      fprintf(stderr, "Time step is too large, exiting\n");
      exit(0);
    }
  }

  /* Naive version may be fine */
  static void testError(int err) {
    if(err != 0) {
      fprintf(stderr, "Time step is too large, exiting\n");
      exit(0);
    }
  }

  // Internal functions
  KOKKOS_INLINE_FUNCTION
  static void lag3_basis(double posx, double coef[3]) {
    coef[0] = .5 * (posx - 1.) * (posx - 2.);
    coef[1] = -1. * (posx) * (posx - 2.);
    coef[2] = .5 * (posx) * (posx - 1.);
  }
   
  KOKKOS_INLINE_FUNCTION
  static void lag5_basis(double posx, double coef[5]) {
    double const loc[] = {1. / 24., -1. / 6., 1. / 4., -1. / 6., 1. / 24.};
   
    coef[0] = loc[0] * (posx - 1.) * (posx - 2.) * (posx - 3.) * (posx - 4.);
    coef[1] = loc[1] * (posx) * (posx - 2.) * (posx - 3.) * (posx - 4.);
    coef[2] = loc[2] * (posx) * (posx - 1.) * (posx - 3.) * (posx - 4.);
    coef[3] = loc[3] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 4.);
    coef[4] = loc[4] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 3.);
  } 

  #if LAG_ORDER == 4
    KOKKOS_INLINE_FUNCTION
    static void lag_basis(float64 posx, float64 coef[LAG_PTS]) {
      const float64 loc[] = {1. / 24., -1. / 6., 1. / 4., -1. / 6., 1. / 24.};
      
      coef[0] = loc[0] * (posx - 1.) * (posx - 2.) * (posx - 3.) * (posx - 4.);
      coef[1] = loc[1] * (posx) * (posx - 2.) * (posx - 3.) * (posx - 4.);
      coef[2] = loc[2] * (posx) * (posx - 1.) * (posx - 3.) * (posx - 4.);
      coef[3] = loc[3] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 4.);
      coef[4] = loc[4] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 3.);
    }
  #endif

  #if LAG_ORDER == 5
    KOKKOS_INLINE_FUNCTION
    static void lag_basis(float64 px, float64 coef[LAG_PTS]) {
      const float64 loc[] = {-1. / 24, 1. / 24., -1. / 12., 1. / 12., -1. / 24., 1. / 24.};
      const float64 pxm2 = px - 2.;
      const float64 sqrpxm2 = pxm2 * pxm2;
      const float64 pxm2_01 = pxm2 * (pxm2 - 1.);
      
      coef[0] = loc[0] * pxm2_01 * (pxm2 + 1.) * (pxm2 - 2.) * (pxm2 - 1.);
      coef[1] = loc[1] * pxm2_01 * (pxm2 - 2.) * (5 * sqrpxm2 + pxm2 - 8.);
      coef[2] = loc[2] * (pxm2 - 1.) * (pxm2 - 2.) * (pxm2 + 1.) * (5 * sqrpxm2 - 3 * pxm2 - 6.);
      coef[3] = loc[3] * pxm2 * (pxm2 + 1.) * (pxm2 - 2.) * (5 * sqrpxm2 - 7 * pxm2 - 4.);
      coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
      coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);
    }
  #endif

  #if LAG_ORDER == 7
    KOKKOS_INLINE_FUNCTION
    static void lag_basis(float64 px, float64 coef[LAG_PTS]) {
      const float64 loc[] = {-1. / 720, 1. / 720, -1. / 240, 1. / 144, -1. / 144, 1. / 240, -1. / 720, 1. / 7    20};
      const float64 pxm3 = px - 3.;
      const float64 sevenpxm3sqr = 7. * pxm3 * pxm3;
      const float64 f1t3 = (px - 1.) * (px - 2.) * (px - 3.);
      const float64 f4t6 = (px - 4.) * (px - 5.) * (px - 6.);
      
      coef[0] = loc[0] * f1t3 * f4t6 * (px - 4.);
      coef[1] = loc[1] * (px - 2.) * pxm3 * f4t6 * (sevenpxm3sqr + 8. * pxm3 - 18.);
      coef[2] = loc[2] * (px - 1.) * pxm3 * f4t6 * (sevenpxm3sqr + 2. * pxm3 - 15.);
      coef[3] = loc[3] * (px - 1.) * (px - 2.) * f4t6 * (sevenpxm3sqr - 4. * pxm3 - 12.);
      coef[4] = loc[4] * f1t3 * (px - 5.) * (px - 6.) * (sevenpxm3sqr - 10. * pxm3 - 9.);
      coef[5] = loc[5] * f1t3 * (px - 4.) * (px - 6.) * (sevenpxm3sqr - 16. * pxm3 - 6.);
      coef[6] = loc[6] * f1t3 * (px - 4.) * (px - 5.) * (sevenpxm3sqr - 22. * pxm3 - 3.);
      coef[7] = loc[7] * f1t3 * f4t6 * pxm3;
    }
  #endif

  KOKKOS_INLINE_FUNCTION
  void truncatedSize(int tile_size[], int pos[], int defaultTileSize[], int maxVal[]) {
    const auto sx = min(maxVal[0] - pos[0], defaultTileSize[0]);
    const auto sy = min(maxVal[1] - pos[1], defaultTileSize[1]);
    
    tile_size[0] = sx;
    tile_size[1] = sy;
  }

  struct blocked_advect_2D_xy_functor {
    Config *conf_;
    RealView4D fn_;
    RealView4D fn_tmp_;
    Kokkos::Experimental::ScatterView<int*> scatter_error_;
    //int vxmax_, vymax_;
    //int ts_vx_, ts_vy_;
    float64 dt_;
    float64 minPhyx_, minPhyy_;
    float64 inv_dx_, inv_dy_;
    int default_tile_size_[2];
    int max_nvx_nvy_[2];
    int local_start_[4];
    int local_end_[4]; 
    float64 minPhy_[4];
    float64 dx_[4];

    blocked_advect_2D_xy_functor(Config *conf, RealView4D fn, RealView4D fn_tmp, float64 dt, Kokkos::Experimental::ScatterView<int*> scatter_error)
      : conf_(conf), fn_(fn), fn_tmp_(fn_tmp), dt_(dt), scatter_error_(scatter_error) {
      const Domain *dom = &(conf->dom_);
      for(int k = 0; k < DIMENSION; k++) {
        local_start_[k] = dom->local_nxmin_[k];
        local_end_[k]   = dom->local_nxmax_[k];
        minPhy_[k]      = dom->minPhy_[k];
        dx_[k]          = dom->dx_[k];
      }
      minPhyx_ = dom->minPhy_[0];
      minPhyy_ = dom->minPhy_[1];
      inv_dx_  = 1. / dom->dx_[0];
      inv_dy_  = 1. / dom->dx_[1];
    }

    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    int interp_2D(const ViewType &tmp2d, const float64 xstar, const float64 ystar, float64 &interp) const {
      #ifdef LAG_ORDER
        #ifdef LAG_ODD
          int ipos1 = floor((xstar - minPhyx_) * inv_dx_);
          int ipos2 = floor((ystar - minPhyy_) * inv_dy_);
        #else
          int ipos1 = round((xstar - minPhyx_) * inv_dx_);
          int ipos2 = round((ystar - minPhyy_) * inv_dy_);
        #endif
        const float64 d_prev1 = LAG_OFFSET + inv_dx_ * (xstar - (minPhyx_ + ipos1 * inv_dx_));
        const float64 d_prev2 = LAG_OFFSET + inv_dy_ * (ystar - (minPhyy_ + ipos2 * inv_dy_));
        
        float64 coefx[LAG_PTS];
        float64 coefy[LAG_PTS];
        float64 ftmp = 0.;

        ipos1 -= LAG_OFFSET;
        ipos2 -= LAG_OFFSET;
        lag_basis(d_prev1, coefx);
        lag_basis(d_prev2, coefy);

        if(ipos1 < local_start_[0] - HALO_PTS || ipos1 > local_end_[0] + HALO_PTS - LAG_ORDER)
          return 1;
        if(ipos2 < local_start_[1] - HALO_PTS || ipos2 > local_end_[1] + HALO_PTS - LAG_ORDER)
          return 1;

        for(int k2 = 0; k2 <= LAG_ORDER; k2++)
          for(int k1 = 0; k1 <= LAG_ORDER; k1++)
            int jx = ipos1 + k1 - local_start_[0] + HALO_PTS;
            int jy = ipos2 + k2 - local_start_[1] + HALO_PTS;
            ftmp += coefx[k1] * coefy[k2] * tmp2d(jx, jy);
        interp = ftmp;
        return 0;
      #else
        const float64 fx = (xstar - minPhyx_) * inv_dx_;
        const float64 fy = (ystar - minPhyy_) * inv_dy_;
        const int ix = floor(fx);
        const int iy = floor(fy);
        const float64 wx = fx - static_cast<float64>(ix);
        const float64 wy = fy - static_cast<float64>(iy);
         
        const float64 etax3 = (1. / 6.) * wx * wx * wx;
        const float64 etax0 = (1. / 6.) + .5 * wx * (wx - 1.) - etax3;
        const float64 etax2 = wx + etax0 - 2. * etax3;
        const float64 etax1 = 1. - etax0 - etax2 - etax3;
        const float64 etax[4] = {etax0, etax1, etax2, etax3};
        
        const float64 etay3 = (1. / 6.) * wy * wy * wy;
        const float64 etay0 = (1. / 6.) + .5 * wy * (wy - 1.) - etay3;
        const float64 etay2 = wy + etay0 - 2. * etay3;
        const float64 etay1 = 1. - etay0 - etay2 - etay3;
        const float64 etay[4] = {etay0, etay1, etay2, etay3};
        
        if(ix < local_start_[0] - 1 || ix > local_end_[0])
          return 1;
        if(iy < local_start_[1] - 1 || iy > local_end_[1])
          return 1;
        float64 ftmp = 0.;

        for(int jy = 0; jy <= 3; jy++) {
          float64 sum = 0.;
          for(int jx = 0; jx <= 3; jx++) {
            sum += etax[jx] * tmp2d(ix - 1 + jx - local_start_[0] + HALO_PTS, 
                                    iy - 1 + jy - local_start_[1] + HALO_PTS);
          }
          ftmp += etay[jy] * sum;
        }
        interp = ftmp;
        return 0;
      #endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int iy, const int ivx) const {
      const int nvy = local_end_[3] - local_start_[3] + 1;

      int err = 0;
      for(int ivy = 0; ivy < nvy; ivy++) {
        int jvx = ivx + local_start_[2];
        int jvy = ivy + local_start_[3];

        const float64 vx = minPhy_[2] + jvx * dx_[2];
        const float64 vy = minPhy_[3] + jvy * dx_[3];
        const float64 depx = dt_ * vx;
        const float64 depy = dt_ * vy;

        auto tmp2d = Kokkos::subview(fn_tmp_, Kokkos::ALL, Kokkos::ALL, ivx + HALO_PTS, ivy + HALO_PTS);

        int jx = ix + local_start_[0];
        int jy = iy + local_start_[1];

        const float64 x = minPhy_[0] + jx * dx_[0];
        const float64 y = minPhy_[1] + jy * dx_[1];

        const float64 xstar = x - depx;
        const float64 ystar = y - depy;
        float64 ftmp = 0;

        auto access_error = scatter_error_.access();
        access_error(0) += interp_2D(tmp2d, xstar, ystar, ftmp);
        fn_(ix  + HALO_PTS, 
            iy  + HALO_PTS,
            ivx + HALO_PTS,
            ivy + HALO_PTS) = ftmp;
      }
    }
  };

  struct advect_4D_functor {
    Config *conf_;
    Efield *ef_;
    RealView4D fn_;
    RealView4D fn_tmp_;
    RealView2D ex_;
    RealView2D ey_;
    Kokkos::Experimental::ScatterView<int*> scatter_error_;
    float64 rxmin_[4], rxmax_[4];
    float64 rxwidth_[4], inv_dx_[4], dx_[4];
    float64 locrxmindx_[4], locrxmaxdx_[4], locrxmin_[4], locrxmax_[4];
    float64 dt_;
    int local_start_[4];
    int local_end_[4];
    int nxmax_[4];

    advect_4D_functor(Config *conf, Efield *ef, RealView4D fn, RealView4D fn_tmp, float64 dt, Kokkos::Experimental::ScatterView<int*> scatter_error)
      : conf_(conf), ef_(ef), fn_(fn), fn_tmp_(fn_tmp), scatter_error_(scatter_error), dt_(dt) {
      const Domain *dom = &(conf->dom_);

      for(int k = 0; k < DIMENSION; k++) {
        nxmax_[k]       = dom->nxmax_[k];
        local_start_[k] = dom->local_nxmin_[k];
        local_end_[k]   = dom->local_nxmax_[k];
        rxmin_[k] = dom->minPhy_[k];
        rxmax_[k] = dom->maxPhy_[k];
        // local minimum in this MPI process
        locrxmin_[k] = dom->minPhy_[k] + local_start_[k] * dom->dx_[k];
        // local maximum in this MPI process
        locrxmax_[k] = dom->minPhy_[k] + local_end_[k] * dom->dx_[k];

        locrxmindx_[k] = locrxmin_[k] - dom->dx_[k];
        locrxmaxdx_[k] = locrxmax_[k] + dom->dx_[k];
        rxwidth_[k]    = dom->maxPhy_[k] - dom->minPhy_[k];
        inv_dx_[k]     = 1. / dom->dx_[k];
        dx_[k]         = dom->dx_[k];
      }

      ex_ = ef_->ex_;
      ey_ = ef_->ey_;
    }

    KOKKOS_INLINE_FUNCTION
    void computeFeet(float64 xstar[4], int pos []) const {
      const int s_nxmax = nxmax_[0];
      const int s_nymax = nxmax_[1];

      const float64 x  = rxmin_[0] + pos[0] * dx_[0];
      const float64 y  = rxmin_[1] + pos[1] * dx_[1];
      const float64 vx = rxmin_[2] + pos[2] * dx_[2];
      const float64 vy = rxmin_[3] + pos[3] * dx_[3];

      float64 xtmp[4];
      xtmp[0] = x  - 0.5 * dt_ * vx;
      xtmp[1] = y  - 0.5 * dt_ * vy;
      xtmp[2] = vx - 0.5 * dt_ * ex_(pos[0], pos[1]);
      xtmp[3] = vy - 0.5 * dt_ * ey_(pos[0], pos[1]);

      float64 ftmp1 = 0., ftmp2 = 0.;

      for(int count = 0; count < 1; count++) {
        int ipos[2];
        float64 coefx[2][3];

        for(int j = 0; j <= 1; j++) {
          xtmp[j] = inv_dx_[j] * (xtmp[j] - rxmin_[j]);
          ipos[j] = round(xtmp[j]) - 1;
          const float64 posx = xtmp[j] - ipos[j];
          lag3_basis(posx, coefx[j]);
        }

        ftmp1 = 0.;
        ftmp2 = 0.;

        for(int k1 = 0; k1 <= 2; k1++) {
          for(int k0 = 0; k0 <= 2; k0++) {
            int jx = (s_nxmax + ipos[0] + k0) % s_nxmax;
            int jy = (s_nymax + ipos[1] + k1) % s_nymax;
            ftmp1 += coefx[0][k0] * coefx[1][k1] * ex_(jx, jy);
            ftmp2 += coefx[0][k0] * coefx[1][k1] * ey_(jx, jy);
          }
        }

        xtmp[2] = vx - 0.5 * dt_ * ftmp1;
        xtmp[3] = vy - 0.5 * dt_ * ftmp2;
        xtmp[0] = x  - 0.5 * dt_ * xtmp[2];
        xtmp[1] = y  - 0.5 * dt_ * xtmp[3];
      }

      xstar[0] = x - dt_ * xtmp[2];
      xstar[1] = y - dt_ * xtmp[3];
      xstar[2] = max(min(vx - dt_ * ftmp1, rxmin_[2] + rxwidth_[2]), rxmin_[2]);
      xstar[3] = max(min(vy - dt_ * ftmp2, rxmin_[3] + rxwidth_[3]), rxmin_[3]);
    }

    KOKKOS_INLINE_FUNCTION
    float64 interp_4D(const RealView4D &tmp_fn, float64 xstar[]) const {
      int ipos[4];

      for(int j = 0; j < 4; j++) {
        xstar[j] = inv_dx_[j] * (xstar[j] - rxmin_[j]);
      }

      #ifdef LAG_ORDER
        float64 coefx0[LAG_PTS];
        float64 coefx1[LAG_PTS];
        float64 coefx2[LAG_PTS];
        float64 coefx3[LAG_PTS];
        
        ipos[0] = floor(xstar[0]) - LAG_OFFSET;
        ipos[1] = floor(xstar[1]) - LAG_OFFSET;
        ipos[2] = floor(xstar[2]) - LAG_OFFSET;
        ipos[3] = floor(xstar[3]) - LAG_OFFSET;
        lab_basis((xstar[0] - ipos[0]), coefx0);
        lab_basis((xstar[1] - ipos[1]), coefx1);
        lab_basis((xstar[2] - ipos[2]), coefx2);
        lab_basis((xstar[3] - ipos[3]), coefx3);

        float64 ftmp1 = 0.;
        for(int k3 = 0; k3 <= LAG_ORDER; k3++) {
          for(int k2 = 0; k2 <= LAG_ORDER; k2++) {
            float64 ftmp2 = 0.;
            
            for(int k1 = 0; k1 <= LAG_ORDER; k1++) {
              float64 ftmp3 = 0.;
           
              for(int k0 = 0; k0 <= LAG_ORDER; k0++) {
                int jx  = ipos[0] + k0 - local_start_[0] + HALO_PTS;
                int jy  = ipos[1] + k1 - local_start_[1] + HALO_PTS;
                int jvx = ipos[2] + k2 - local_start_[2] + HALO_PTS;
                int jvy = ipos[3] + k3 - local_start_[3] + HALO_PTS;
                ftmp3 += coefx0[k0] * tmp_fn(jx, jy, jvx, jvy);
              }
              ftmp2 += ftmp3 * coefx1[k1];
            }
          }
          ftmp1 += ftmp2 * coefx2[k2] * coefx3[k3];
        }
         
        return ftmp1;
      #else // LAG_ORDER
        float64 eta[4][4];

        for(int j = 0; j < 4; j++) {
          ipos[j] = floor(xstar[j]);
          const float64 wx = xstar[j] - ipos[j];
          const float64 etax3 = (1./6.) * wx * wx * wx;
          const float64 etax0 = (1./6.) + 0.5 * wx * (wx - 1.) - etax3;
          const float64 etax2 = wx + etax0 - 2. * etax3;
          const float64 etax1 = 1. - etax0 - etax2 - etax3;
          eta[j][0] = etax0;
          eta[j][1] = etax1;
          eta[j][2] = etax2;
          eta[j][3] = etax3;
        }

        float64 ftmp1 = 0.;
        for(int k3 = 0; k3 <= 3; k3++) {
          for(int k2 = 0; k2 <= 3; k2++) {
            float64 ftmp2 = 0.;
             
            for(int k1 = 0; k1 <= 3; k1++) {
              float64 ftmp3 = 0.;
              for(int k0 = 0; k0 <= 3; k0++) {
                int jx  = ipos[0] + k0 - 1 - local_start_[0] + HALO_PTS;
                int jy  = ipos[1] + k1 - 1 - local_start_[1] + HALO_PTS;
                int jvx = ipos[2] + k2 - 1 - local_start_[2] + HALO_PTS;
                int jvy = ipos[3] + k3 - 1 - local_start_[3] + HALO_PTS;
                ftmp3 += eta[0][k0] * tmp_fn(jx, jy, jvx, jvy);
              }
              ftmp2 += ftmp3 * eta[1][k1];
            }
            ftmp1 += ftmp2 * eta[2][k2] * eta[3][k3];
          }
        }
        return ftmp1;
      #endif
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int iy, const int ivx) const {
      const int nvy = local_end_[3] - local_start_[3] + 1;
      auto access_error = scatter_error_.access();
      for(int ivy = 0; ivy < nvy; ivy++) {
        int jx  = ix  + local_start_[0];
        int jy  = iy  + local_start_[1];
        int jvx = ivx + local_start_[2];
        int jvy = ivy + local_start_[3];

        float64 xstar[4];
        int indices[4] = {jx, jy, jvx, jvy};
        computeFeet(xstar, indices);

        for(int j = 0; j < 4; j++) {
          access_error(0) += (xstar[j] <  locrxmindx_[j] || xstar[j] > locrxmaxdx_[j]);
        }

        fn_(ix  + HALO_PTS,
            iy  + HALO_PTS,
            ivx + HALO_PTS, 
            ivy + HALO_PTS) = interp_4D(fn_tmp_, xstar);
      }
    }
  };

  void advect_2D_xy(Config *conf, RealView4D fn, float64 dt) {
    int nx  = fn.extent(0);
    int ny  = fn.extent(1);
    int nvx = fn.extent(2);
    int nvy = fn.extent(3);
    RealView4D fn_tmp = RealView4D("fn_tmp", nx, ny, nvx, nvy);
    Impl::deep_copy(fn_tmp, fn);

    int nx_inner  = nx  - HALO_PTS*2;
    int ny_inner  = ny  - HALO_PTS*2;
    int nvx_inner = nvx - HALO_PTS*2;

    MDPolicyType_3D advect_2d_policy3d({{0, 0, 0}},
                                       {{nx_inner, ny_inner, nvx_inner}},
                                       {{TILE_SIZE0, TILE_SIZE1, TILE_SIZE2}}
                                      );
    
    View1D<int> error("error", 1);
    auto scatter_error = Kokkos::Experimental::create_scatter_view(error);

    Kokkos::parallel_for("advect_2d", advect_2d_policy3d, blocked_advect_2D_xy_functor(conf, fn, fn_tmp, dt, scatter_error));
    Kokkos::Experimental::contribute(error, scatter_error);
    testError(error);
  }

  void advect_4D(Config *conf, Efield *ef, RealView4D fn, RealView4D tmp_fn, float64 dt) {
    int nx  = fn.extent(0);
    int ny  = fn.extent(1);
    int nvx = fn.extent(2);
    Kokkos::deep_copy(tmp_fn, fn);

    int nx_inner  = nx  - HALO_PTS*2;
    int ny_inner  = ny  - HALO_PTS*2;
    int nvx_inner = nvx - HALO_PTS*2;
    MDPolicyType_3D advect_4d_policy3d({{0, 0, 0}},
                                       {{nx_inner, ny_inner, nvx_inner}},
                                       {{TILE_SIZE0, TILE_SIZE1, TILE_SIZE2}}
                                      );

    View1D<int> error("error", 1);
    auto scatter_error = Kokkos::Experimental::create_scatter_view(error);
    Kokkos::parallel_for("advect_4d", advect_4d_policy3d, advect_4D_functor(conf, ef, fn, tmp_fn, dt, scatter_error));
    testError(error);
  }

  void print_fxvx(Config *conf, Distrib &comm, const RealView4D fn, int iter) {
    const Domain *dom = &(conf->dom_);
    const int nx  = fn.extent(0);
    const int ny  = fn.extent(1);
    const int nvx = fn.extent(2);
    const int nvy = fn.extent(3);
    int nx_inner  = nx  - HALO_PTS*2;
    int ny_inner  = ny  - HALO_PTS*2;
    int nvx_inner = nvx - HALO_PTS*2;
    int nvy_inner = nvy - HALO_PTS*2;
    RealView2D fnxvx("fnxvx", nx_inner, nvx_inner);
    RealView2D fnxvxres("fnxvxres", nx_inner, nvx_inner);

    RealView2D fnyvy("fnyvy", ny_inner, nvy_inner);
    RealView2D fnyvyres("fnyvyres", ny_inner, nvy_inner);

    // At (iy, ivy) = (0, nvy/2) cross section
    const int iy  = 0;
    const int ivy = dom->nxmax_[3] / 2;
    if (dom->local_nxmin_[3] <= ivy && ivy <= dom->local_nxmax_[3] &&
        dom->local_nxmin_[1] <= iy  && iy  <= dom->local_nxmax_[1]) {
      for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          int jx  = ix - dom->local_nxmin_[0] + HALO_PTS;
          int jy  = iy - dom->local_nxmin_[1] + HALO_PTS;
          int jvx = iy - dom->local_nxmin_[2] + HALO_PTS;
          int jvy = iy - dom->local_nxmin_[3] + HALO_PTS;
          fnxvx(ix, ivx) = fn(jx, jy, jvx, jvy);
        }
      }
    }

    // At (ix, ivx) = (0, nvx/2) cross section
    const int ix  = 0;
    const int ivx = dom->nxmax_[2] / 2;
    if (dom->local_nxmin_[2] <= ivx && ivx <= dom->local_nxmax_[2] &&
        dom->local_nxmin_[0] <= ix  && ix  <= dom->local_nxmax_[0]) {
      for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
        for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
          int jx  = ix - dom->local_nxmin_[0] + HALO_PTS;
          int jy  = iy - dom->local_nxmin_[1] + HALO_PTS;
          int jvx = iy - dom->local_nxmin_[2] + HALO_PTS;
          int jvy = iy - dom->local_nxmin_[3] + HALO_PTS;
          fnyvy(iy, ivy) = fn(jx, jy, jvx, jvy);
        }
      }
    }

    int nelems = dom->nxmax_[0] * dom->nxmax_[2];
    MPI_Reduce(fnxvx.data(), fnxvxres.data(), nelems, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    nelems = dom->nxmax_[1] * dom->nxmax_[3];
    MPI_Reduce(fnyvy.data(), fnyvyres.data(), nelems, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(comm.master()) {
      char filename[128];
      printf("print_fxvx %d\n", iter);
      
      {
        sprintf(filename, "fxvx%04d.out", iter);
        FILE* fileid = fopen(filename, "w");
      
        for(int ivx = 0; ivx < dom->nxmax_[2]; ivx++) {
          for(int ix = 0; ix < dom->nxmax_[0]; ix++)
            fprintf(fileid, "%.7f %.7f %20.13le\n",
                    dom->minPhy_[0] + ix  * dom->dx_[0],
                    dom->minPhy_[2] + ivx * dom->dx_[2],
                    fnxvxres(ix, ivx));
      
          fprintf(fileid, "\n");
        }
        fclose(fileid);
      }
      
      {
        sprintf(filename, "fyvy%04d.out", iter);
        FILE* fileid = fopen(filename, "w");
      
        for(int ivy = 0; ivy < dom->nxmax_[3]; ivy++) {
          for(int iy = 0; iy < dom->nxmax_[1]; iy++)
            fprintf(fileid, "%.7f %.7f %20.13le\n",
                    dom->minPhy_[1] + iy  * dom->dx_[1],
                    dom->minPhy_[3] + ivy * dom->dx_[3],
                    fnyvyres(iy, ivy));
       
          fprintf(fileid, "\n");
        }
        fclose(fileid);
      }
    }
  }

};

#endif
