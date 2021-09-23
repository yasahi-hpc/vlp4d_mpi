#include <numeric>
#include <execution>
#include <algorithm>
#include "Field.hpp"

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

void field_rho(Config *conf, RealView4D &fn, Efield *ef) {
  const Domain *dom = &(conf->dom_);

  int nx_min = dom->local_nxmin_[0],     ny_min = dom->local_nxmin_[1],     nvx_min = dom->local_nxmin_[2],     nvy_min = dom->local_nxmin_[3];
  int nx_max = dom->local_nxmax_[0] + 1, ny_max = dom->local_nxmax_[1] + 1, nvx_max = dom->local_nxmax_[2] + 1, nvy_max = dom->local_nxmax_[3] + 1;
  const int nx = nx_max - nx_min;
  const int ny = ny_max - ny_min;

  float64 dvx = dom->dx_[2], dvy = dom->dx_[3];

  float64 *ptr_fn      = fn.data();
  float64 *ptr_rho_loc = ef->rho_loc_.data();
  auto idx_4d = fn.index();
  auto idx_2d = ef->rho_loc_.index();

  // Define accessors by macro
  #define _fn(j0, j1, j2, j3) ptr_fn[idx_4d(j0, j1, j2, j3)]
  #define _rho_loc(j0, j1) ptr_rho_loc[idx_2d(j0, j1)]

  Coord<2, array_layout::value> coord_2d({nx, ny}, {nx_min, ny_min});

  auto integral = [=](const int idx) {
    int ptr_idx[2];
    coord_2d.to_coord(idx, ptr_idx);
    int ix = ptr_idx[0], iy = ptr_idx[1];
    float64 sum = 0.;
    for(int ivy=nvy_min; ivy<nvy_max; ivy++) {
      for(int ivx=nvx_min; ivx<nvx_max; ivx++) {
        sum += _fn(ix, iy, ivx, ivy);
      }
    }
    _rho_loc(ix, iy) = sum * dvx * dvy;
  };

  std::for_each_n(std::execution::par_unseq,
                  counting_iterator(0), nx*ny,
                  integral);
}

void field_reduce(Config *conf, Efield *ef) {
  // reduction in velocity space
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nelems = nx * ny;

  float64 *ptr_rho     = ef->rho_.data();
  float64 *ptr_rho_loc = ef->rho_loc_.data();
  MPI_Allreduce(ptr_rho_loc, ptr_rho, nelems, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter) {
  const Domain *dom = &(conf->dom_);
  const int nx = dom->nxmax_[0];
  const int ny = dom->nxmax_[1];
  const float64 minPhyx = dom->minPhy_[0];
  const float64 minPhyy = dom->minPhy_[1];
  const float64 dx = dom->dx_[0];
  const float64 dy = dom->dx_[1];

  float64 *ptr_rho = ef->rho_.data();
  float64 *ptr_ex  = ef->ex_.data();
  float64 *ptr_ey  = ef->ey_.data();
  auto idx_2d = ef->rho_loc_.index();

  // Define accessors by macro
  #define _ex(j0, j1) ptr_ex[idx_2d(j0, j1)]
  #define _ey(j0, j1) ptr_ex[idx_2d(j0, j1)]
  #define _rho(j0, j1) ptr_rho[idx_2d(j0, j1)]

  Coord<2, array_layout::value> coord_2d(nx, ny);

  switch(dom->idcase_) {
    case 2:
      std::for_each_n(std::execution::par_unseq,
                      counting_iterator(0), nx*ny,
                      [=](const int idx) {
                        int ptr_idx[2];
                        coord_2d.to_coord(idx, ptr_idx);
                        int ix = ptr_idx[0], iy = ptr_idx[1];
                        _ex(ix, iy) = -(minPhyx + ix * dx);
                        _ey(ix, iy) = 0.;
                      });
      break;
    case 6:
      std::for_each_n(std::execution::par_unseq,
                      counting_iterator(0), nx*ny,
                      [=](const int idx) {
                        int ptr_idx[2];
                        coord_2d.to_coord(idx, ptr_idx);
                        int ix = ptr_idx[0], iy = ptr_idx[1];
                        _ex(ix, iy) = 0.;
                        _ey(ix, iy) = -(minPhyy + iy * dy);
                      });
      break;

    case 10:
    case 20:
      std::for_each_n(std::execution::par_unseq,
                      counting_iterator(0), nx*ny,
                      [=](const int idx) {
                        int ptr_idx[2];
                        coord_2d.to_coord(idx, ptr_idx);
                        int ix = ptr_idx[0], iy = ptr_idx[1];
                        _rho(ix, iy) -= 1.;
                      });

      lu_solve_poisson(conf, ef, dg, iter);
      break;

    default:
      lu_solve_poisson(conf, ef, dg, iter);
      break;
  }
}

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter) {
  const Domain *dom = &(conf->dom_);
  ef->solve_poisson_fftw(dom->maxPhy_[0], dom->maxPhy_[1]);
  dg->compute(conf, ef, iter);
};
