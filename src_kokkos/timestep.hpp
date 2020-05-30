#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "efield.hpp"
#include "diags.hpp"
#include "types.h"
#include "Math.hpp"
#include "communication.hpp"
#include "Advection.hpp"
#include "Timer.hpp"
#include "helper.hpp"

void onetimestep(Config *conf, Distrib &comm, RealView4D fn, RealView4D fnp1, Efield *ef, Diags *dg, std::vector<Timer*> &timers, int iter);

void onetimestep(Config *conf, Distrib &comm, RealView4D fn, RealView4D fnp1, Efield *ef, Diags *dg, std::vector<Timer*> &timers, int iter) {
  Domain *dom = &(conf->dom_);

  // Exchange halo of the local domain in order to perform
  // the advection afterwards (the interpolation needs the
  // extra points located in the halo region)
  comm.exchangeHalo(conf, fn, timers);

  timers[Splinecoeff_xy]->begin();
  Spline::computeCoeff_xy(conf, fn);
  Impl::deep_copy(fnp1, fn);
  Kokkos::fence();
  timers[Splinecoeff_xy]->end();

  timers[Advec2D]->begin();
  Advection::advect_2D_xy(conf, fn, 0.5 * dom->dt_);
  Kokkos::fence();
  timers[Advec2D]->end();

  timers[Field]->begin();
  field_rho(conf, comm, fn, ef);
  field_poisson(conf, ef, dg, iter);
  Kokkos::fence();
  timers[Field]->end();

  timers[Splinecoeff_vxvy]->begin();
  Spline::computeCoeff_vxvy(conf, fnp1);
  Kokkos::fence();
  timers[Splinecoeff_vxvy]->end();

  timers[Advec4D]->begin();
  Advection::advect_4D(conf, ef, fnp1, fn, dom->dt_);
  Kokkos::fence();
  timers[Advec4D]->end();

  timers[Field]->begin();
  field_rho(conf, comm, fnp1, ef);
  field_poisson(conf, ef, dg, iter);
  Kokkos::fence();
  timers[Field]->end();

  timers[Diag]->begin();
  dg->computeL2norm(conf, fnp1, iter);

  if(iter % dom->ifreq_ == 0) {
    if(dom->fxvx_) Advection::print_fxvx(conf, comm, fnp1, iter);
    dg->save(conf, comm, iter);
  }
  Kokkos::fence();
  timers[Diag]->end();
};

#endif
