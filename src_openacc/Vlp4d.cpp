#include "Types.hpp"
#include "Communication.hpp"
#include "Timer.hpp"
#include "Efield.hpp"
#include "Diags.hpp"
#include "Field.hpp"
#include "Parser.hpp"
#include "Init.hpp"
#include "Timestep.hpp"

int main(int argc, char *argv[]) {
  Parser parser(argc, argv);
  Distrib comm(argc, argv);

  Config conf;
  RealView4D fn, fnp1;
  Efield *ef = nullptr;
  Diags  *dg = nullptr;

  std::vector<Timer*> timers;
  defineTimers(timers);

  // Initialization
  printf("reading input file %s\n", parser.file_);
  init(parser.file_, &conf, comm, fn, fnp1, &ef, &dg, timers);
  int iter = 0;
  field_rho(&conf, fn, ef);
  field_reduce(&conf, ef);
  field_poisson(&conf, ef, dg, iter);
  dg->computeL2norm(&conf, fn, iter);

  // Main time step loop
  while(iter < conf.dom_.nbiter_) {
    if(comm.master()) {
      printf("iter %d\n", iter);
    }

    iter++;
    onetimestep(&conf, comm, fn, fnp1, ef, dg, timers, iter);
    fn.swap(fnp1);
  }

  if(comm.master()) {
    printTimers(timers);
  }

  freeTimers(timers);
  comm.finalize();
  return 0;
}