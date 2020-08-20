#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include "OpenACC_View.hpp"

static void L2norm(RealView2D &view, const int rank) {
  double norm;

  shape_nd<2> shape = view.strides();
  shape_nd<2> offset = view.offsets();

  int n0 = shape[0], n1 = shape[1];
  int n0_start = offset[0], n1_start = offset[1];

  view.updateSelf();

  for(int iy = 0; iy < n1; iy++) {
    for(int ix = 0; ix < n0; ix++) {
      int jx = ix + n0_start;
      int jy = iy + n1_start;
      norm += view(jx, jy) * view(jx, jy);
    }
  }

  std::stringstream ss;
  ss << "L2 norm of 2 dimensional view " << view.name() << " @ rank " << rank << ": " << std::scientific << std::setprecision(15) << norm;
  std::cout << ss.str() << std::endl;
}

static void L2norm(RealView4D &view, const int rank) {
  double norm;

  shape_nd<4> shape = view.strides();
  shape_nd<4> offset = view.offsets();

  int n0 = shape[0], n1 = shape[1], n2 = shape[2], n3 = shape[3];
  int n0_start = offset[0], n1_start = offset[1], n2_start = offset[2], n3_start = offset[3];

  view.updateSelf();
  for(int ivy = 0; ivy < n3; ivy++) {
    for(int ivx = 0; ivx < n2; ivx++) {
      for(int iy = 0; iy < n1; iy++) {
        for(int ix = 0; ix < n0; ix++) {
          int jx  = ix  + n0_start;
          int jy  = iy  + n1_start;
          int jvx = ivx + n2_start;
          int jvy = ivy + n3_start;
                                    
          norm += view(jx, jy, jvx, jvy) * view(jx, jy, jvx, jvy);
        }
      }
    }
  }

  std::stringstream ss;
  ss << "L2 norm of 4 dimensional view " << view.name() << " @ rank " << rank << ": " << std::scientific << std::setprecision(15) << norm;
  std::cout << ss.str() << std::endl;
}

#endif
