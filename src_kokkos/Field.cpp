#include "Field.hpp"
#include "index.h"
#include "helper.hpp"
#include "tiles.h"

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

void field_rho(Config *conf, Distrib &comm, RealView4D fn, Efield *ef)
{
  const Domain *dom = &(conf->dom_);

  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  float64 dvx = dom->dx_[2], dvy = dom->dx_[3];

  // Capturing a class member causes a problem
  // See https://github.com/kokkos/kokkos/issues/695
  RealView2D rho     = ef->rho_;
  RealView2D rho_loc = ef->rho_loc_;
  RealView2D ex      = ef->ex_;
  RealView2D ey      = ef->ey_;
  RealView2D phi     = ef->phi_;

  MDPolicyType_2D initialization_policy2d({{0, 0}},
                                          {{nx, ny}},
                                          {{TILE_SIZE0, TILE_SIZE1}}
                                         );

  Kokkos::parallel_for("initialization", initialization_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
    rho(ix, iy)     = 0.;
    rho_loc(ix, iy) = 0.;
    ex(ix, iy)      = 0.;
    ey(ix, iy)      = 0.;
    phi(ix, iy)     = 0.;
  });

  int nx_loc  = dom->local_nx_[0];
  int ny_loc  = dom->local_nx_[1];
  int nvx_loc = dom->local_nx_[2];
  int nvy_loc = dom->local_nx_[3];
  int local_xstart  = dom->local_nxmin_[0];
  int local_ystart  = dom->local_nxmin_[1];
  int local_vxstart = dom->local_nxmin_[2];
  int local_vystart = dom->local_nxmin_[3];
  MDPolicyType_2D integral_policy2d({{0, 0}},
                                    {{nx_loc, ny_loc}},
                                    {{BASIC_TILE_SIZE0, BASIC_TILE_SIZE1}}
                                   );

  Kokkos::parallel_for("integral", integral_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
    float64 sum = 0.;
    for(int ivy=0; ivy<nvy_loc; ivy++) {
      for(int ivx=0; ivx<nvx_loc; ivx++) {
        sum += fn(ix  + HALO_PTS,
                  iy  + HALO_PTS,
                  ivx + HALO_PTS,
                  ivy + HALO_PTS);
      }
    }

    // stored in the global address
    rho_loc(ix + local_xstart, iy + local_ystart) = sum * dvx * dvy;
  });

  // reduction in velocity space
  // For some reason, MPI_Allreduce is fairly slow
  int nelems = nx * ny;
  MPI_Allreduce(rho_loc.data(), rho.data(), nelems, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
};

void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter)
{
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  double dx = dom->dx_[0], dy = dom->dx_[1];
  double minPhyx = dom->minPhy_[0], minPhyy = dom->minPhy_[1];

  RealView2D rho = ef->rho_;
  RealView2D ex  = ef->ex_;
  RealView2D ey  = ef->ey_;

  MDPolicyType_2D poisson_policy2d({{0, 0}},
                                   {{nx, ny}},
                                   {{BASIC_TILE_SIZE0, BASIC_TILE_SIZE1}}
                                  );
  switch(dom->idcase_)
  {
    case 2:
        Kokkos::parallel_for("poisson", poisson_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
          ex(ix, iy) = -(minPhyx + ix * dx);
          ey(ix, iy) = 0.;
        });
        break;
    case 6:
        Kokkos::parallel_for("poisson", poisson_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
          ey(ix, iy) = -(minPhyy + iy * dy);
          ex(ix, iy) = 0.;
        });
        break;
    case 10:
    case 20:
        Kokkos::parallel_for("poisson", poisson_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
          rho(ix, iy) -= 1.;
        });
        lu_solve_poisson(conf, ef, dg, iter);
        break;
    default:
        lu_solve_poisson(conf, ef, dg, iter);
        break;
  }
};

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter)
{
  const Domain *dom = &(conf->dom_);
  ef->solve_poisson_fftw(dom->maxPhy_[0], dom->maxPhy_[1]);
  dg->compute(conf, ef, iter);
};
