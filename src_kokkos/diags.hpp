#ifndef __DIAGS_HPP__
#define __DIAGS_HPP__

#include "types.h"
#include "config.h"
#include "efield.hpp"
#include "communication.hpp"

struct Diags
{
private:
  typedef RealView1D::HostMirror RealHostView1D;
  RealHostView1D nrj_;
  RealHostView1D nrjx_;
  RealHostView1D nrjy_;
  RealHostView1D mass_;
  RealHostView1D l2norm_;
  int last_iter_ = 0;

public:
  Diags(Config *conf);
  virtual ~Diags();

  void compute(Config *conf, Efield *ef, int iter);
  void computeL2norm(Config *conf, RealOffsetView4D fn, int iter);
  void save(Config *conf, Distrib &comm, int cur_iter);
};

#endif
