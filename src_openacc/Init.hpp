#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Config.hpp"
#include "Communication.hpp"
#include "Transpose.hpp"

void init(const char *file, Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, Efield **ef, Diags **dg, Impl::Transpose<float64, array_layout::value> **transpose, std::vector<Timer*> &timers);
void finalize(Efield **ef, Diags **dg, Impl::Transpose<float64, array_layout::value> **transpose);

#endif
