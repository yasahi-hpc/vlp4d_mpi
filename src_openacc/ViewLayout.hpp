#ifndef __VIEWLAYOUT_HPP__
#define __VIEWLAYOUT_HPP__

enum class Layout {LayoutLeft, LayoutRight};

#if defined ( LAYOUT_LEFT )
  typedef std::integral_constant<Layout, Layout::LayoutLeft> array_layout;
#elif defined ( LAYOUT_RIGHT )
  typedef std::integral_constant<Layout, Layout::LayoutRight> array_layout;
#else
  #if defined( ENABLE_OPENACC )
    // Layout left for OpenACC
    typedef std::integral_constant<Layout, Layout::LayoutLeft> array_layout;
  #else
    // Layout right for OpenMP
    typedef std::integral_constant<Layout, Layout::LayoutRight> array_layout;
  #endif
#endif

#endif
