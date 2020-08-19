#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

/* 
 * Halo exchange on f^n
 * Compute spline coeff. along x, y: f^n -> f_xy^n
 * Advection 2D in space x, y of f^n
 * Compute density f_xy^n -> rho^n+1/2 and solve Poisson in Fourier space phi^n+1/2
 * Diagnostics/outputs on phi^n+1/2
 *
 * Compute spline coeff. along vx, vy: f_xy^n -> f_xy,vxvy^n
 * Estimate 4D displacements: phi^n+1/2 -> delta^n+1/2
 * Advection 4D in x, y, vx, vy of f^n: f_xy,vxvy^n and delta^n+1/2 -> f^n+1
 *
 */

#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Communication.hpp"
#include "Spline.hpp"
#include "Advection.hpp"
#include "Config.hpp"
#include "Timer.hpp"
#include "Math.hpp"

void onetimestep(Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1,
                 Efield *ef, Diags *dg, std::vector<Timer*> &timers, int iter);

void onetimestep(Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1,
                 Efield *ef, Diags *dg, std::vector<Timer*> &timers, int iter) {
  Domain *dom = &(conf->dom_);

  // Exchange halo of the local domain in order to perform
  // the advection afterwards (the interpolation needs the 
  // extra points located in the halo region)
  comm.exchangeHalo(conf, fn, timers); // OK

  Spline::computeCoeff_xy(conf, fn); // May be OK
  Impl::deep_copy(fnp1, fn);
  Advection::advect_2D_xy(conf, fn, 0.5 * dom->dt_);

  field_rho(conf, fn, ef); // OK
  field_reduce(conf, ef); // OK
  field_poisson(conf, ef, dg, iter); //OK

  std::cout << "before spline vxvy" << std::endl;
  Spline::computeCoeff_vxvy(conf, fnp1); // May be OK
  std::cout << "after spline vxvy" << std::endl;
  Advection::advect_4D(conf, ef, fnp1, fn, dom->dt_);
  field_rho(conf, fnp1, ef); // OK
  field_reduce(conf, ef); // OK
  field_poisson(conf, ef, dg, iter); // OK

  dg->computeL2norm(conf, fnp1, iter); // OK

  if(iter % dom->ifreq_ == 0) {
    if(dom->fxvx_) Advection::print_fxvx(conf, comm, fnp1, iter);
    dg->save(conf, comm, iter);
  }

}


#endif
