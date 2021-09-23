#include "Diags.hpp"
#include "Index.hpp"
#include <omp.h>
#include <numeric>
#include <execution>
#include <algorithm>

// This may be useful
// https://docs.microsoft.com/en-us/cpp/parallel/concrt/how-to-perform-map-and-reduce-operations-in-parallel?view=msvc-160

Diags::Diags(Config *conf) {
  const Domain *dom = &(conf->dom_);
  const int nx = dom->nxmax_[0];
  const int ny = dom->nxmax_[1];
  const int nbiter = dom->nbiter_ + 1;
  nrj_    = RealView1D("nrj",  nbiter);
  nrjx_   = RealView1D("nrjx", nbiter);
  nrjy_   = RealView1D("nrjy", nbiter);
  mass_   = RealView1D("mass", nbiter);
  l2norm_ = RealView1D("l2norm", nbiter);
  //vars_.resize(nx*ny, 0);
}

void Diags::compute(Config *conf, Efield *ef, int iter) {
  const Domain *dom = &conf->dom_;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);
  
  /*
  float64 iter_mass = 0.;
  float64 it_nrj = 0., it_nrjx = 0., it_nrjy = 0.;

  float64 *ptr_ex = ef->ex_.data(), *ptr_ey = ef->ey_.data();
  float64 *ptr_rho = ef->rho_.data();

  auto* ptr_vars = vars_.data();
  auto moment = [=](const int jxy) {
    const float64 eex = ptr_ex[jxy];
    const float64 eey = ptr_ey[jxy];
    ptr_vars[jxy].x = ptr_rho[jxy];
    ptr_vars[jxy].y = eex * eex + eey * eey;
    ptr_vars[jxy].z = eex * eex;
    ptr_vars[jxy].w = eey * eey;
  };

  std::for_each_n(std::execution::par_unseq,
                  counting_iterator(0), nx*ny,
                  moment);

  auto reduced_vars = std::reduce(
    std::execution::par_unseq, vars_.begin(), vars_.end());

  iter_mass = reduced_vars.x;
  it_nrj    = reduced_vars.y;
  it_nrjx   = reduced_vars.z;
  it_nrjy   = reduced_vars.w;
  */

  float64 iter_mass = std::reduce(std::execution::par_unseq, ef->rho_.vector().begin(), ef->rho_.vector().end());
  float64 it_nrjx = std::transform_reduce(
    std::execution::par_unseq, ef->ex_.vector().begin(), ef->ex_.vector().end(), ef->ex_.vector().begin(), 0.0);
  float64 it_nrjy = std::transform_reduce(
    std::execution::par_unseq, ef->ey_.vector().begin(), ef->ey_.vector().end(), ef->ey_.vector().begin(), 0.0);
  float64 it_nrj = it_nrjx + it_nrjy;

  it_nrj = sqrt(it_nrj * dom->dx_[0] * dom->dx_[1]);
  it_nrj = it_nrj > 1.e-30 ? log(it_nrj) : -1.e9;

  nrj_(iter)  = it_nrj;
  nrjx_(iter) = sqrt(0.5 * it_nrjx * dom->dx_[0] * dom->dx_[1]);
  nrjy_(iter) = sqrt(0.5 * it_nrjy * dom->dx_[0] * dom->dx_[1]);
  mass_(iter) = iter_mass * dom->dx_[0] * dom->dx_[1];
}

void Diags::computeL2norm(Config *conf, RealView4D &fn, int iter) {
  const Domain *dom = &conf->dom_;
  int nx_min = dom->local_nxmin_[0], ny_min = dom->local_nxmin_[1], nvx_min = dom->local_nxmin_[2], nvy_min = dom->local_nxmin_[3];
  int nx_max = dom->local_nxmax_[0], ny_max = dom->local_nxmax_[1], nvx_max = dom->local_nxmax_[2], nvy_max = dom->local_nxmax_[3];
  const int nx  = nx_max  - nx_min + 1;
  const int ny  = ny_max  - ny_min + 1;
  const int nvx = nvx_max - nvx_min + 1;
  const int nvy = nvy_max - nvy_min + 1;

  const int n = nx * ny * nvx * nvy;
  RealView1D l2("l2", n);
  Coord<4, array_layout::value> coord_4d({nx, ny, nvx, nvy}, {nx_min, ny_min, nvx_min, nvy_min});
  float64 *ptr_l2 = l2.data();
  float64 *ptr_fn = fn.data();
  auto idx_4d = fn.index();

  // Define accessors by macro
  #define _fn(j0, j1, j2, j3) ptr_fn[idx_4d(j0, j1, j2, j3)]
  #define _l2(j0) ptr_l2[j0]

  auto l2_norm = [=](int idx) {
    int ptr_idx[4];
    coord_4d.to_coord(idx, ptr_idx);
    int ix = ptr_idx[0], iy = ptr_idx[1], ivx = ptr_idx[2], ivy = ptr_idx[3];
    _l2(idx) = _fn(ix, iy, ivx, ivy) * _fn(ix, iy, ivx, ivy);
  };

  std::for_each_n(std::execution::par_unseq,
                  counting_iterator(0), n,
                  l2_norm);

  float64 l2loc = std::reduce(std::execution::par_unseq, l2.vector().begin(), l2.vector().end());

  float64 l2glob = 0.;
  MPI_Reduce(&l2loc, &l2glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  l2norm_(iter) = sqrt(l2glob * dom->dx_[0] * dom->dx_[1] * dom->dx_[2] * dom->dx_[3]);
}

void Diags::save(Config *conf, Distrib &comm, int cur_iter) {
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
