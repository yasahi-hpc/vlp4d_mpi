#ifndef __VIEWLAYOUT_HPP__
#define __VIEWLAYOUT_HPP__

enum class Layout {LayoutLeft, LayoutRight};

#if defined ( LAYOUT_LEFT )
  using array_layout = std::integral_constant<Layout, Layout::LayoutLeft>;
#elif defined ( LAYOUT_RIGHT )
  using array_layout = std::integral_constant<Layout, Layout::LayoutRight>;
#else
  #if defined( ENABLE_OPENMP_OFFLOAD )
    // Layout left for OpenMP offload
    using array_layout = std::integral_constant<Layout, Layout::LayoutLeft>;
  #else
    // Layout right for OpenMP
    using array_layout = std::integral_constant<Layout, Layout::LayoutRight>;
  #endif
#endif

#endif
