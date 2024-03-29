#ifndef __INDEX_HPP__
#define __INDEX_HPP__

#include <Kokkos_Core.hpp>

namespace Index {
  KOKKOS_INLINE_FUNCTION 
  int coord_2D2int(int i1, int i2, int n1, int n2) {
    int idx = i1 + n1*i2;
    return idx;
  }

  KOKKOS_INLINE_FUNCTION 
  int coord_3D2int(int i1, int i2, int i3, int n1, int n2, int n3) {
    int idx = i1 + i2*n1 + i3*n1*n2;
    return idx;
  }
  
  KOKKOS_INLINE_FUNCTION 
  int coord_4D2int(int i1, int i2, int i3, int i4, int n1, int n2, int n3, int n4) {
    int idx = i1 + i2*n1 + i3*n1*n2 + i4*n1*n2*n3;
    return idx;
  }
};

#endif
