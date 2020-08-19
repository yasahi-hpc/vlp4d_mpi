#ifndef __MATH_HPP__
#define __MATH_HPP__

#if defined ( ENABLE_OPENACC ) 
  #include <openacc.h>
#else
  #include <omp.h>
#endif

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
    typedef typename ViewType::value_type_ type;
    type *ptr_a = a.data();
    type *ptr_b = b.data();
    
    #if defined ( ENABLE_OPENACC ) 
      #pragma acc data present(ptr_a, ptr_b)
      #pragma acc parallel loop
    #else
      #pragma omp parallel for
    #endif
    for(int i = 0; i < n; i++) {
      ptr_a[i] = ptr_b[i];
    }
  }
};

#endif
