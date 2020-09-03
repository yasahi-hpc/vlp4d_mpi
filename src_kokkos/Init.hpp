#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Config.hpp"
#include "communication.hpp"

void init(const char *file, Config *conf, Distrib &comm, RealOffsetView4D &fn, RealOffsetView4D &fnp1, Efield **ef, Diags **dg, std::vector<Timer*> &timers);
void finalize(Efield **ef, Diags **dg);

#endif
