#ifndef __INIT_H__
#define __INIT_H__

#include "efield.hpp"
#include "diags.hpp"
#include "types.h"
#include "config.h"
#include "communication.hpp"

void init(const char *file, Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, Efield **ef, Diags **dg, std::vector<Timer*> &timers);
void finalize(Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, Efield **ef, Diags **dg);


#endif
