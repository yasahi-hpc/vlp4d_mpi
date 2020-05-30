#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include "types.h"
#include "index.h"

/*
template <class ViewType>
static void L2norm(ViewType &view) {
  const size_t rank = view.rank;
  typedef typename ViewType::value_type value_type;

  value_type norm;

  if(rank == 2) {

  } else if (rank == 3) {
    const size_t n0 = view.extent_int(0);
    const size_t n1 = view.extent_int(1);
    const size_t n2 = view.extent_int(2);
    Kokkos::parallel_reduce(n0 * n1 * n2, KOKKOS_LAMBDA (const int i, double& lsum) {
      int3 idx_3D = Index::int2coord_3D(i, n0, n1, n2);
      int ix = idx_3D.x, iy = idx_3D.y, ivx = idx_3D.z;
      lsum += view(ix, iy, ivx) * view(ix, iy, ivx);
    }, norm);
  } else if (rank == 4) {
    const size_t n0 = view.extent_int(0);
    const size_t n1 = view.extent_int(1);
    const size_t n2 = view.extent_int(2);
    const size_t n3 = view.extent_int(3);
    Kokkos::parallel_reduce(n0 * n1 * n2 * n3, KOKKOS_LAMBDA (const int i, double& lsum) {
      int4 idx_4D = Index::int2coord_4D(i, n0, n1, n2);
      int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;
      lsum += view(ix, iy, ivx, ivy) * view(ix, iy, ivx, ivy);
    }, norm);
  }
  
  std::stringstream ss;
  ss << "L2 norm of rank " << rank << " view " << view.label() << ": " << std::scientific << std::setprecision(15) << norm;
}
*/

template <typename T>
static void L2norm(View2D<T> &view, int rank) {
  const size_t dim = view.rank;
  T norm;

  const size_t n0 = view.extent_int(0);
  const size_t n1 = view.extent_int(1);
  Kokkos::parallel_reduce(n0 * n1, KOKKOS_LAMBDA (const int i, double& lsum) {
    int2 idx_2D = Index::int2coord_2D(i, n0, n1);
    int ix = idx_2D.x, iy = idx_2D.y;
    lsum += view(ix, iy) * view(ix, iy);
  }, norm);

  std::stringstream ss;
  ss << "L2 norm of " << dim << " dimensional view " << view.label() << " @ rank " << rank << ": " << std::scientific << std::setprecision(15) << norm;
  std::cout << ss.str() << std::endl;
}

template <typename T>
static void printAll(View2D<T> &view, int rank) {

  const size_t n0 = view.extent_int(0);
  const size_t n1 = view.extent_int(1);

  std::stringstream ss;
  ss << view.label() << ".rank" << rank << ".dat";

  std::ofstream outfile;
  outfile.open(ss.str(), std::ios::out);

  typename View2D<T>::HostMirror h_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(h_view, view);

  for(int i1 = 0; i1 < n1; i1++) {
    for(int i0 = 0; i0 < n0; i0++) {
      outfile << h_view(i0,i1) << ", ";
    }
    outfile << "\n";
  }
}

template <typename T>
static void printAll(LeftView2D<T> &view, int rank) {
  T norm;

  const size_t n0 = view.extent_int(0);
  const size_t n1 = view.extent_int(1);

  std::stringstream ss;
  ss << view.label() << ".rank" << rank << ".dat";

  std::ofstream outfile;
  outfile.open(ss.str(), std::ios::out);

  typename LeftView2D<T>::HostMirror h_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(h_view, view);

  for(int i1 = 0; i1 < n1; i1++) {
    for(int i0 = 0; i0 < n0; i0++) {
      outfile << h_view(i0,i1) << ", ";
    }
    outfile << "\n";
  }
}

template <typename T>
static void L2norm(LeftView2D<T> &view, int rank) {
  const size_t dim = view.rank;
  T norm;

  const size_t n0 = view.extent_int(0);
  const size_t n1 = view.extent_int(1);
  Kokkos::parallel_reduce(n0 * n1, KOKKOS_LAMBDA (const int i, double& lsum) {
    int2 idx_2D = Index::int2coord_2D(i, n0, n1);
    int ix = idx_2D.x, iy = idx_2D.y;
    lsum += view(ix, iy) * view(ix, iy);
  }, norm);

  // Compute on host
  typename LeftView2D<T>::HostMirror h_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(h_view, view);

  norm = 0;
  for(int i = 0; i < n0*n1; i++) {
    int2 idx_2D = Index::int2coord_2D(i, n0, n1);
    int ix = idx_2D.x, iy = idx_2D.y;
    norm += h_view(ix, iy) * h_view(ix, iy);
  }

  std::stringstream ss;
  ss << "L2 norm of " << dim << " dimensional view " << view.label() << " @ rank " << rank << ": " << std::scientific << std::setprecision(15) << norm;
  std::cout << ss.str() << std::endl;
}

template <typename T>
static void L2norm(View4D<T> &view, int rank) {
  const size_t dim = view.rank;
  T norm;

  const size_t n0 = view.extent_int(0);
  const size_t n1 = view.extent_int(1);
  const size_t n2 = view.extent_int(2);
  const size_t n3 = view.extent_int(3);
  Kokkos::parallel_reduce(n0 * n1 * n2 * n3, KOKKOS_LAMBDA (const int i, double& lsum) {
    int4 idx_4D = Index::int2coord_4D(i, n0, n1, n2, n3);
    int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;
    lsum += view(ix, iy, ivx, ivy) * view(ix, iy, ivx, ivy);
  }, norm);

  typename View4D<T>::HostMirror h_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(h_view, view);
  norm = 0;
  for(int i = 0; i < n0*n1*n2*n3; i++) {
    int4 idx_4D = Index::int2coord_4D(i, n0, n1, n2, n3);
    int ix = idx_4D.x, iy = idx_4D.y, ivx = idx_4D.z, ivy = idx_4D.w;
    norm += h_view(ix, iy, ivx, ivy) * h_view(ix, iy, ivx, ivy);
  }

  std::stringstream ss;
  ss << "L2 norm of " << dim << " dimensional view " << view.label() << " @ rank " << rank << ": " << std::scientific << std::setprecision(15) << norm;
  std::cout << ss.str() << std::endl;
}

#endif
