#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "Config.hpp"
#include "Efield.hpp"
#include "Diags.hpp"
#include "types.h"
#include "communication.hpp"

void field_rho(Config *conf, Distrib &comm, RealOffsetView4D fn, Efield *ef);
void field_reduce(Config *conf, Efield *ef);
void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

#endif
