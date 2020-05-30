#include "diags.hpp"
#include "index.h"
#include "helper.hpp"
#include "tiles.h"
#include <Kokkos_ScatterView.hpp>
#include <mpi.h>
#include <stdio.h>

Diags::Diags(Config *conf)
{
  const int nbiter = conf->dom_.nbiter_ + 1;

  nrj_    = RealHostView1D("nrj",    nbiter);
  nrjx_   = RealHostView1D("nrjx",   nbiter);
  nrjy_   = RealHostView1D("nrjy",   nbiter);
  mass_   = RealHostView1D("mass",   nbiter);
  l2norm_ = RealHostView1D("l2norm", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter)
{
  const Domain *dom = &conf->dom_;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);
  RealView2D ex  = ef->ex_; 
  RealView2D ey  = ef->ey_; 
  RealView2D rho = ef->rho_; 

  // Capturing a class member causes a problem
  // See https://github.com/kokkos/kokkos/issues/695
  //
  /*
  float64 mass;
  Scalar2<float64> ee;
  MDPolicyType_2D moment_policy2d({{0, 0}},
                                  {{nx, ny}},
                                  {{TILE_SIZE0, TILE_SIZE1}}
                                 );
  Kokkos::parallel_reduce("moments", moment_policy2d, KOKKOS_LAMBDA (const int &ix, const int &iy, float64 &lsums) {
    lsums += rho(ix, iy);
  }, mass);

  Kokkos::parallel_reduce("moments", moment_policy2d, KOKKOS_LAMBDA (const int &ix, const int &iy, Scalar2<float64> &lsums) {
    const float64 eex = ex(ix, iy);
    const float64 eey = ey(ix, iy);

    lsums.x += eex * eex;
    lsums.y += eey * eey;
  }, ee);

  float64 iter_mass = mass;
  float64 it_nrj    = ee.x * ee.y;
  float64 it_nrjx   = ee.x;
  float64 it_nrjy   = ee.y;
  */


  //typedef Kokkos::View<float64[4], execution_space> ScalarsView;
  //ScalarsView sums;
  /*
  Scalar4<float64> sums; sums.x = 0; sums.y = 0; sums.z = 0; sums.w = 0;
  MDPolicyType_2D moment_policy2d({{0, 0}},
                                  {{nx, ny}},
                                  {{TILE_SIZE0, TILE_SIZE1}}
                                 );

  Kokkos::parallel_reduce("moments", moment_policy2d, KOKKOS_LAMBDA (const int &ix, const int &iy, Scalar4<float64> &lsums) {
    const float64 eex = ex(ix, iy);
    const float64 eey = ey(ix, iy);

    lsums.x += rho(ix, iy);
    lsums.y += eex * eex + eey * eey;
    lsums.z += eex * eex;
    lsums.w += eey * eey;
  }, sums);
  */
  /*
  double_3 sums; sums.x = 0; sums.y = 0; sums.z = 0;
  MDPolicyType_2D moment_policy2d({{0, 0}},
                                  {{nx, ny}},
                                  {{TILE_SIZE0, TILE_SIZE1}}
                                 );

  Kokkos::parallel_reduce("moments", moment_policy2d, KOKKOS_LAMBDA (const int &ix, const int &iy, double_3 &lsums) {
    const float64 eex = ex(ix, iy);
    const float64 eey = ey(ix, iy);

    lsums.x += rho(ix, iy);
    lsums.y += eex * eex + eey * eey;
    lsums.z += eex * eex;
  }, sums);
  */

  //typename ScalarsView::HostMirror h_sums  = Kokkos::create_mirror_view(sums);
  //Kokkos::deep_copy(h_sums, sums);
  //
  // Use Scatter view for reduction
  typedef Kokkos::View<float64*, execution_space> ScalarsView;
  ScalarsView sums("sum", 4);
  auto scatter_sums = Kokkos::Experimental::create_scatter_view(sums);
  MDPolicyType_2D moment_policy2d({{0, 0}},
                                  {{nx, ny}},
                                  {{TILE_SIZE0, TILE_SIZE1}}
                                 );

  Kokkos::parallel_for("moments", moment_policy2d, KOKKOS_LAMBDA (const int &ix, const int &iy) {
    const float64 eex = ex(ix, iy);
    const float64 eey = ey(ix, iy);

    auto access_sums = scatter_sums.access();
    access_sums(0) += rho(ix, iy);
    access_sums(1) += eex * eex + eey * eey;
    access_sums(2) += eex * eex;
    access_sums(3) += eey * eey;
  });

  Kokkos::Experimental::contribute(sums, scatter_sums);
  typename ScalarsView::HostMirror h_sums = Kokkos::create_mirror_view(sums);
  Kokkos::deep_copy(h_sums, sums);
  float64 iter_mass = h_sums(0);
  float64 it_nrj    = h_sums(1);
  float64 it_nrjx   = h_sums(2);
  float64 it_nrjy   = h_sums(3);

  it_nrj = sqrt(it_nrj * dom->dx_[0] * dom->dx_[1]);
  it_nrj = it_nrj > 1.e-30 ? log(it_nrj) : -1.e9;

  nrj_(iter)  = it_nrj;
  nrjx_(iter) = sqrt(0.5 * it_nrjx * dom->dx_[0] * dom->dx_[1]);
  nrjy_(iter) = sqrt(0.5 * it_nrjy * dom->dx_[0] * dom->dx_[1]);
  mass_(iter) = iter_mass * dom->dx_[0] * dom->dx_[1];
}

void Diags::computeL2norm(Config *conf, RealView4D fn, int iter)
{
  const Domain *dom = &conf->dom_;
  int nx = dom->local_nx_[0], ny = dom->local_nx_[1], nvx = dom->local_nx_[2], nvy = dom->local_nx_[3];

  // Capturing a class member causes a problem
  // See https://github.com/kokkos/kokkos/issues/695
  float64 l2loc = 0.;
  MDPolicyType_4D moment_policy4d({{0, 0, 0, 0}},
                                  {{nx, ny, nvx, nvy}},
                                  {{TILE_SIZE0, TILE_SIZE1, TILE_SIZE2, TILE_SIZE3}}
                                 );
  Kokkos::parallel_reduce("l2norm", moment_policy4d, KOKKOS_LAMBDA (const int &ix, const int &iy, const int &ivx, const int &ivy, float64 &lsum) {
    int jx  = ix  + HALO_PTS;
    int jy  = iy  + HALO_PTS;
    int jvx = ivx + HALO_PTS;
    int jvy = ivy + HALO_PTS;
    lsum += fn(jx, jy, jvx, jvy) * fn(jx, jy, jvx, jvy);
  }, l2loc);

  float64 l2glob = 0;
  MPI_Reduce(&l2loc, &l2glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  l2norm_(iter) = sqrt(l2glob * dom->dx_[0] * dom->dx_[1] * dom->dx_[2] * dom->dx_[3]);
}

void Diags::save(Config *conf, Distrib &comm, int cur_iter)
{
  const Domain* dom = &conf->dom_;

  char filename[16];

  if(comm.master()) {
    {
      sprintf(filename, "nrj.out");

      FILE *fileid = fopen(filename, (last_iter_ == 0 ? "w": "a"));
      for(int iter=last_iter_; iter<= cur_iter; ++iter)
        fprintf(fileid, "%17.13e %17.13e %17.13e %17.13e %17.13e\n", iter * dom->dt_, nrj_(iter), nrjx_(iter), nrjy_(iter), mass_(iter));

      fclose(fileid);
    }

    {
      sprintf(filename, "l2norm.out");

      FILE *fileid = fopen(filename, (last_iter_ == 0 ? "w": "a"));
      for(int iter=last_iter_; iter<= cur_iter; ++iter)
        fprintf(fileid, "%17.13e %17.13e\n", iter * dom->dt_, l2norm_(iter));

      fclose(fileid);
    }
  }
}

Diags::~Diags() {};
