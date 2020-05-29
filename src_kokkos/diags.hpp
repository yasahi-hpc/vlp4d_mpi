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
  void computeL2norm(Config *conf, RealView4D fn, int iter);
  void save(Config *conf, Distrib &comm, int cur_iter);
};


/*
struct Moment {
  Config *conf_;
  Efield *ef_;
  int s_nxmax_, s_nymax_;

  Moment(Config *conf, Efield *ef)
    : conf_(conf), ef_(ef) {
    const Domain *dom = &(conf_->dom_);
    s_nxmax  = dom->nxmax_[0];
    s_nymax  = dom->nxmax_[1];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ix, const int iy, )
};
*/

#endif
