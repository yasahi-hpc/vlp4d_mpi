#ifndef __DIAGS_HPP__
#define __DIAGS_HPP__

#include "Types.hpp"
#include "Config.hpp"
#include "Efield.hpp"
#include "Communication.hpp"

template <typename ScalarType>
struct MomentType {
  ScalarType x, y, z, w;
};

struct Diags {
private:
  using moment_type = MomentType<float64>;
  RealView1D nrj_;
  RealView1D nrjx_;
  RealView1D nrjy_;
  RealView1D mass_;
  RealView1D l2norm_;
  std::vector<moment_type> vars_;
  int last_iter_ = 0;

public:
  Diags(Config *conf);
  ~Diags(){}

  void compute(Config *conf, Efield *ef, int iter);
  void computeL2norm(Config *conf, RealView4D &fn, int iter);
  void save(Config *conf, Distrib &comm, int cur_iter);
};

#endif
