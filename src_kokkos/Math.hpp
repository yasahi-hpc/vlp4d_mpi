#ifndef __MATH_HPP__
#define __MATH_HPP__

#include <Kokkos_Core.hpp>

namespace Impl {

  template <class ViewType>
  void swap(ViewType &a, ViewType &b) {
    ViewType tmp = a;
    a = b;
    b = tmp;
  }

  template <class ViewType>
  void deep_copy(ViewType &a, ViewType &b) {
    const size_t n = a.size();
    Kokkos::parallel_for("copy", n, KOKKOS_LAMBDA(const int i) {
      a.data()[i] = b.data()[i];
    });
  }

  template <class ViewType>
  void free(ViewType &a) {
    a = ViewType();
  }
};

#endif
