#ifndef __VIEWLAYOUT_HPP__
#define __VIEWLAYOUT_HPP__

enum class Layout {LayoutLeft, LayoutRight};

#if defined ( LAYOUT_LEFT )
  typedef std::integral_constant<Layout, Layout::LayoutLeft> array_layout;
#elif defined ( LAYOUT_RIGHT )
  typedef std::integral_constant<Layout, Layout::LayoutRight> array_layout;
#else
  #if defined( _NVHPC_STDPAR_GPU )
    // Layout left for GPU execution
    /* -stdpar=gpu turns on CUDA support without defining __CUDACC__.
     * But __CUDACC__ needs to be defined when <cuda_runtime.h> is included.
     * see. _cuda_preinclude.h
     */
    typedef std::integral_constant<Layout, Layout::LayoutLeft> array_layout;
  #else
    // Layout right for OpenMP
    typedef std::integral_constant<Layout, Layout::LayoutRight> array_layout;
  #endif
#endif

#endif
