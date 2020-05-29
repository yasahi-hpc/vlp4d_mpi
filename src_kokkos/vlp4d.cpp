/*
 * @brief The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). 
 *        From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. 
 *        Vlasov solver is typically based on a directional Strang splitting. 
 *        The Poisson equation is treated with 2D Fourier transforms. 
 *        For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions.
 *        The Vlasov solver is based on advection's operators:
 *
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *        Poisson solver -> compute electric fields Ex and E
 *        1D advection along vx (Dt)
 *        1D advection along vy (Dt)
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *
 *        Interpolation operator within advection is Lagrange polynomial of order 5, 7 depending on a compilation flag (order 5 by default).
 *
 *  @author
 *  @url    https://gitlab.maisondelasimulation.fr/GyselaX/vlp4d/tree/master
 */

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include "helper.hpp"
#include "types.h"
#include "config.h"
#include "init.h"
#include "parser.hpp"
#include "communication.hpp"
#include "efield.hpp"
#include "diags.hpp"
#include "Field.hpp"
#include "Math.hpp"
#include "spline.hpp"
#include "timestep.hpp"
#include "Timer.hpp"

int main (int argc, char* argv[]) {
  Parser parser;
  parser.setArgs(argc, argv);
  Distrib comm(argc, argv);

  std::vector<Timer*> timers;
  defineTimers(timers);

  // When initializing Kokkos, you may pass in command-line arguments,
  // just like with MPI_Init().  Kokkos reserves the right to remove
  // arguments from the list that start with '--kokkos-'.
  Kokkos::InitArguments args_kokkos;
  args_kokkos.num_threads = parser.num_threads_;
  args_kokkos.num_numa    = parser.teams_;
  args_kokkos.device_id   = parser.device_;

  Kokkos::initialize (args_kokkos);
  {
    Config conf;
    RealView4D fn, fnp1;
    Efield *ef = NULL;
    Diags *dg = NULL;

    // Initialization
    printf("reading input file %s\n", parser.file_);
    init(parser.file_, &conf, comm, fn, fnp1, &ef, &dg, timers);
    int iter = 0;

    Kokkos::fence();
    Kokkos::Timer timer;
    timers[Total]->begin();

    timers[Field]->begin();
    field_rho(&conf, comm, fn, ef);
    field_poisson(&conf, ef, dg, iter);
    dg->computeL2norm(&conf, fn, iter);
    Kokkos::fence();
    timers[Field]->end();

    while(iter <conf.dom_.nbiter_) {
      printf("iter %d\n", iter);

      iter++;
      onetimestep(&conf, comm, fn, fnp1, ef, dg, timers, iter);
      Impl::swap(fn, fnp1);
    }
    Kokkos::fence();
    timers[Total]->end();
    double seconds = timer.seconds();
    printf("total time: %f s\n", seconds);
    finalize(&conf, comm, fn, fnp1, &ef, &dg);
    comm.cleanup();
  }
  Kokkos::finalize();
  if(comm.master()) {
    printTimers(timers);
  }
  freeTimers(timers);
  comm.finalize();
  return 0;
}
