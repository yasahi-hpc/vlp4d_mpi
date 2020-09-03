#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "Config.hpp"
#include "Efield.hpp"
#include "Diags.hpp"
#include "types.h"
#include "communication.hpp"

/*
 * @param[in] fn
 * @param[out] ef.rho_ (Updated by the integral of fn)
 * @param[out] ef.ex_ (zero initialization)
 * @param[out] ef.ey_ (zero initialization)
 * @param[out] ef.phi_ (zero initialization)
 */
void field_rho(Config *conf, Distrib &comm, RealOffsetView4D fn, Efield *ef);
void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

#endif
