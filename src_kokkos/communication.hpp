#ifndef __COMM_H__
#define __COMM_H__

#include <vector>
#include <mpi.h>
#include <algorithm>
#include "types.h"
#include "config.h"
#include "index.h"
#include "utils.hpp"
#include "Timer.hpp"

static constexpr int VUNDEF = -100000;

// Element considered within the algorith of the Unbalanced Recursive Bisection (URB)
struct Urbnode{
  // xmin and xmax set the interval of the box 
  // owned by a specific MPI process
  int xmin_[DIMENSION];
  int xmax_[DIMENSION];

  // Number of processes
  int nbp_;

  // ID of the MPI process ([Y. A comment] duplicated???)
  int pid_;
};

// One single halo part stuck to the local 
// This halo part should be a regular contiguous box without 
// hole inside. This halo part will be then bond to the box
// owned by the local MPI process.
struct Halo{
  int xmin_[DIMENSION];
  int xmax_[DIMENSION];
  int pid_; 
  int size_;
  int lxmin_[DIMENSION];
  int lxmax_[DIMENSION];
  float64 *buf_;
  int tag_;
};

// In Kokkos, we manage all the halos in a single data structure
struct Halos{
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  RealLeftView2D buf_;
  RealView1D buf_flatten_;
  RangeView2D xmin_;
  RangeView2D xmax_;
  RangeView2D bc_in_min_;
  RangeView2D bc_in_max_;
  RangeView2D lxmin_;
  RangeView2D lxmax_;

  //shape_t<DIMENSION> lxmin_;
  //shape_t<DIMENSION> lxmax_;
  shape_t<DIMENSION> nhalo_max_;
  //RangeView2D lxmin_;
  //RangeView2D lxmax_;
  int size_;     // buffer size of each halo
  int nb_halos_; // the number of halos
  std::vector<int> pids_;
  std::vector<int> tags_;

  /* Return the */
  float64* buf(const int i) {
    float64* dptr_buf = buf_flatten_.data();
  }

  void set(std::vector<Halo> &list, const std::string name) {
    nb_halos_ = list.size();
    pids_.resize(nb_halos_);
    tags_.resize(nb_halos_);

    xmin_      = RangeView2D("halo_xmin",  nb_halos_);
    xmax_      = RangeView2D("halo_xmax",  nb_halos_);
    lxmin_     = RangeView2D("halo_lxmin", nb_halos_);
    lxmax_     = RangeView2D("halo_lxmax", nb_halos_);
    bc_in_min_ = RangeView2D("bc_in_min", nb_halos_);
    bc_in_max_ = RangeView2D("bc_in_max", nb_halos_);
    typename RangeView2D::HostMirror h_xmin  = Kokkos::create_mirror_view(xmin_);
    typename RangeView2D::HostMirror h_xmax  = Kokkos::create_mirror_view(xmax_);
    typename RangeView2D::HostMirror h_lxmin = Kokkos::create_mirror_view(lxmin_);
    typename RangeView2D::HostMirror h_lxmax = Kokkos::create_mirror_view(lxmax_);
    typename RangeView2D::HostMirror h_bc_in_min = Kokkos::create_mirror_view(bc_in_min_);
    typename RangeView2D::HostMirror h_bc_in_max = Kokkos::create_mirror_view(bc_in_max_);

    std::vector<int> sizes;
    std::vector<int> nx_halos, ny_halos, nvx_halos, nvy_halos;
    for(size_t i = 0; i < nb_halos_; i++) {
      Halo *halo = &(list[i]);
      int tmp_size = (halo->xmax_[0] - halo->xmin_[0] + 1) * (halo->xmax_[1] - halo->xmin_[1] + 1)
                   * (halo->xmax_[2] - halo->xmin_[2] + 1) * (halo->xmax_[3] - halo->xmin_[3] + 1);
      sizes.push_back(tmp_size);
      nx_halos.push_back(halo->xmax_[0] - halo->xmin_[0] + 1);
      ny_halos.push_back(halo->xmax_[1] - halo->xmin_[1] + 1);
      nvx_halos.push_back(halo->xmax_[2] - halo->xmin_[2] + 1);
      nvy_halos.push_back(halo->xmax_[3] - halo->xmin_[3] + 1);

      pids_[i] = halo->pid_;
      tags_[i] = halo->tag_;

      for(int j = 0; j < DIMENSION; j++) {
        h_xmin(i, j)  = halo->xmin_[j]; 
        h_xmax(i, j)  = halo->xmax_[j];
        h_lxmin(i, j) = halo->lxmin_[j]; 
        h_lxmax(i, j) = halo->lxmax_[j]; 
        int lxmin = h_lxmin(i, j) - HALO_PTS, lxmax = h_lxmax(i, j) + HALO_PTS;
        h_bc_in_min(i, j) = (h_xmin(i, j) <= lxmin && lxmin <= h_xmax(i, j)) ? lxmin : VUNDEF;
        h_bc_in_max(i, j) = (h_xmin(i, j) <= lxmax && lxmax <= h_xmax(i, j)) ? lxmax : VUNDEF;
      }
    }
    Kokkos::deep_copy(xmin_,  h_xmin);
    Kokkos::deep_copy(xmax_,  h_xmax);
    Kokkos::deep_copy(lxmin_, h_lxmin);
    Kokkos::deep_copy(lxmax_, h_lxmax);
    Kokkos::deep_copy(bc_in_min_, h_bc_in_min);
    Kokkos::deep_copy(bc_in_max_, h_bc_in_max);

    // Prepare large enough buffer
    auto max_size = std::max_element(sizes.begin(), sizes.end());
    size_ = *max_size;

    nhalo_max_[0] = *std::max_element(nx_halos.begin(),  nx_halos.end());
    nhalo_max_[1] = *std::max_element(ny_halos.begin(),  ny_halos.end());
    nhalo_max_[2] = *std::max_element(nvx_halos.begin(), nvx_halos.end());
    nhalo_max_[3] = *std::max_element(nvy_halos.begin(), nvy_halos.end());
    //size_ = HALO_PTS * (*max_size) * (*max_size) * (*max_size);

    buf_ = RealLeftView2D(name + "_buffer", size_, nb_halos_);
  }
};

struct Distrib{
private:
  // Pseudo variable
  int NB_DIMS_DECOMPOSITION = 4;

  // ID of the MPI process
  int pid_;

  // Number of MPI processes
  int nbp_;

  // List of boxes representing all local MPI domains
  std::vector<Urbnode> ulist_;

  // List of halo buffers (receiving side)
  std::vector<Halo> recv_list_;
  Halos recv_buffers_;

  // List of halo buffers (sending side)
  std::vector<Halo> send_list_;
  Halos send_buffers_; // May be better to use pointer for explicit deallocation

  // The local box for the very local MPI domain
  Urbnode *node_;

  // Use spline or not
  bool spline_;

  // Global domain size
  int nxmax_[DIMENSION];

public:
  Distrib(int &nargs, char **argv) : spline_(true) {
    int required = MPI_THREAD_SERIALIZED;
    int provided;

    // Initialize MPI
    MPI_Init_thread(&nargs, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp_);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid_);
  };

  ~Distrib(){};
  void cleanup() {
    // deallocate views
  }
  void finalize(){
    MPI_Finalize();
  };

  // Getters
  bool master(){return pid_==0;};
  int pid(){return pid_;};
  int rank(){return pid_;};
  int nbp(){return nbp_;};
  Urbnode *node(){return node_;};
  std::vector<Urbnode> &nodes(){return ulist_;};

  // Initializers
  void createDecomposition(Config *conf);
  void neighboursList(Config *conf, RealView4D halo_fn); 
  void bookHalo(Config *conf);

  // Communication
  void exchangeHalo(Config *conf, RealView4D halo_fn, std::vector<Timer*> &timers);
private:
  void getNeighbours(const Config *conf, const RealView4Dhost halo_fn, int xrange[8],
                     std::vector<Halo> &hlist, int lxmin[4], int lxmax[4], int count);

  void fillHalo(const Config *conf, RealView4D halo_fn);
  void fillHaloBoundaryCond(const Config *conf, RealView4D halo_fn);
  void fillHaloBoundaryCondOrig(const Config *conf, RealView4D halo_fn);

  void applyBoundaryCondition(Config *conf, RealView4D halo_fn);

  int mergeElts(std::vector<Halo> &v, std::vector<Halo>::iterator &f, std::vector<Halo>::iterator &g);

private:

  DISALLOW_DEFAULT(Distrib);
  DISALLOW_COPY_AND_ASSIGN(Distrib);
};

// functors
struct pack {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  Config         *conf_;
  RealView4D     halo_fn_;
  RealLeftView2D buf_;
  Halos          send_halos_;
  RangeView2D    xmin_, xmax_;
  int nx_max_, ny_max_, nvx_max_, nvy_max_;
  int local_xstart_, local_ystart_, local_vxstart_, local_vystart_;

  pack(Config *conf, RealView4D halo_fn, Halos send_halos)
    : conf_(conf), halo_fn_(halo_fn), send_halos_(send_halos) {
    buf_  = send_halos_.buf_;
    xmin_ = send_halos_.xmin_;
    xmax_ = send_halos_.xmax_;
    const Domain *dom = &(conf->dom_);
    nx_max_  = dom->nxmax_[0];
    ny_max_  = dom->nxmax_[1];
    nvx_max_ = dom->nxmax_[2];
    nvy_max_ = dom->nxmax_[3];

    local_xstart_  = dom->local_nxmin_[0];
    local_ystart_  = dom->local_nxmin_[1];
    local_vxstart_ = dom->local_nxmin_[2];
    local_vystart_ = dom->local_nxmin_[3];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ix, const int iy, const int ivx, const int ib) const {
    const int ix_min  = xmin_(ib, 0), ix_max  = xmax_(ib, 0); 
    const int iy_min  = xmin_(ib, 1), iy_max  = xmax_(ib, 1);
    const int ivx_min = xmin_(ib, 2), ivx_max = xmax_(ib, 2);
    const int ivy_min = xmin_(ib, 3), ivy_max = xmax_(ib, 3);

    const int nx  = ix_max  - ix_min  + 1;
    const int ny  = iy_max  - iy_min  + 1;
    const int nvx = ivx_max - ivx_min + 1;
    const int nvy = ivy_max - ivy_min + 1;

    const int jx  = ix  + ix_min;
    const int jy  = iy  + iy_min;
    const int jvx = ivx + ivx_min;
    if ( (jx <= ix_max) && (jy <= iy_max) && (jvx <= ivx_max) ) {
      for(int ivy = 0; ivy < nvy; ivy++) {
        // Pack into halo->buf as a 1D flatten array
        // periodice boundary condition in each direction
        // Always Layout Left
        const int jvy = ivy + ivy_min;
        const int ix_bc  = (nx_max_  + jx)  % nx_max_  - local_xstart_  + HALO_PTS;
        const int iy_bc  = (ny_max_  + jy)  % ny_max_  - local_ystart_  + HALO_PTS;
        const int ivx_bc = (nvx_max_ + jvx) % nvx_max_ - local_vxstart_ + HALO_PTS;
        const int ivy_bc = (nvy_max_ + jvy) % nvy_max_ - local_vystart_ + HALO_PTS;
        int idx = Index::coord_4D2int(ix, iy, ivx, ivy, nx, ny, nvx, nvy);
        buf_(idx, ib) = halo_fn_(ix_bc, iy_bc, ivx_bc, ivy_bc);
      }
    }
  }
};

struct unpack {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  Config         *conf_;
  RealView4D     halo_fn_;
  RealLeftView2D buf_;
  Halos          recv_halos_;
  RangeView2D    xmin_, xmax_;
  int local_xstart_, local_ystart_, local_vxstart_, local_vystart_;

  unpack(Config *conf, RealView4D halo_fn, Halos recv_halos)
    : conf_(conf), halo_fn_(halo_fn), recv_halos_(recv_halos) {
    buf_  = recv_halos_.buf_;
    xmin_ = recv_halos_.xmin_;
    xmax_ = recv_halos_.xmax_;
    const Domain *dom = &(conf->dom_);
    local_xstart_  = dom->local_nxmin_[0];
    local_ystart_  = dom->local_nxmin_[1];
    local_vxstart_ = dom->local_nxmin_[2];
    local_vystart_ = dom->local_nxmin_[3];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ix, const int iy, const int ivx, const int ib) const {
    const int ix_min  = xmin_(ib, 0), ix_max  = xmax_(ib, 0); 
    const int iy_min  = xmin_(ib, 1), iy_max  = xmax_(ib, 1);
    const int ivx_min = xmin_(ib, 2), ivx_max = xmax_(ib, 2);
    const int ivy_min = xmin_(ib, 3), ivy_max = xmax_(ib, 3);

    const int nx  = ix_max  - ix_min  + 1;
    const int ny  = iy_max  - iy_min  + 1;
    const int nvx = ivx_max - ivx_min + 1;
    const int nvy = ivy_max - ivy_min + 1;

    const int jx  = ix  + ix_min;
    const int jy  = iy  + iy_min;
    const int jvx = ivx + ivx_min;
    if ( (jx <= ix_max) && (jy <= iy_max) && (jvx <= ivx_max) ) {
      for(int ivy = 0; ivy < nvy; ivy++) {
        const int jvy = ivy + ivy_min;
        // buf_ is Always Layout Left
        int idx = Index::coord_4D2int(ix, iy, ivx, ivy, nx, ny, nvx, nvy);
        halo_fn_(jx  - local_xstart_  + HALO_PTS, 
                 jy  - local_ystart_  + HALO_PTS, 
                 jvx - local_vxstart_ + HALO_PTS,
                 jvy - local_vystart_ + HALO_PTS
                ) = buf_(idx, ib);
      }
    }
  }
};

/*
  @biref Compute boundary conditions to derive spline coefficients afterwards.
         This algorithm is complex and equivalent to halo_fill_boundary_cond_orig.
         Called in fillHalo
  @param[in] halo_fn
    Indentical to fn
  @param[out] halo
    1D array packing fn
 */
struct boundary_condition {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  Config         *conf_;
  RealView4D     halo_fn_;
  RealLeftView2D buf_;
  Halos          send_halos_;
  RangeView2D    xmin_, xmax_;
  RangeView2D    bc_in_min_, bc_in_max_;

  float64 alpha_;
  // Global domain size
  int nx_, ny_, nvx_, nvy_;

  // Local domain min and max
  int local_start_[4];
  int local_xstart_, local_ystart_, local_vxstart_, local_vystart_;

  // Pseudo constants
  int bc_sign_[8];

  boundary_condition(Config *conf, RealView4D halo_fn, Halos send_halos)
    : conf_(conf), halo_fn_(halo_fn), send_halos_(send_halos) {
    buf_  = send_halos_.buf_;
    xmin_ = send_halos_.xmin_;
    xmax_ = send_halos_.xmax_;
    bc_in_min_ = send_halos_.bc_in_min_;
    bc_in_max_ = send_halos_.bc_in_max_;
    const Domain *dom = &(conf->dom_);
    nx_  = dom->nxmax_[0];
    ny_  = dom->nxmax_[1];
    nvx_ = dom->nxmax_[2];
    nvy_ = dom->nxmax_[3];
    alpha_ = sqrt(3) - 2;

    for(int k = 0; k < DIMENSION; k++) {
      bc_sign_[2 * k + 0] = -1;
      bc_sign_[2 * k + 1] = 1;
      local_start_[k] = dom->local_nxmin_[k];
    }

    // Without halo region
    local_xstart_  = dom->local_nxmin_[0];
    local_ystart_  = dom->local_nxmin_[1];
    local_vxstart_ = dom->local_nxmin_[2];
    local_vystart_ = dom->local_nxmin_[3];
  }

  // For test purpose, parallelized over ib only
  KOKKOS_INLINE_FUNCTION
  void operator()(const int ib) const {
    int halo_min[4], halo_max[4];
    for(int k = 0; k < DIMENSION; k++) {
      halo_min[k] = xmin_(ib, k);
      halo_max[k] = xmax_(ib, k);
    }

    const int halo_nx  = halo_max[0] - halo_min[0] + 1;
    const int halo_ny  = halo_max[1] - halo_min[1] + 1;
    const int halo_nvx = halo_max[2] - halo_min[2] + 1;
    const int halo_nvy = halo_max[3] - halo_min[3] + 1;

    int bc_in[8], orcheck[4];

    int orcsum = 0;
    char bitconf = 0;
    for(int k = 0; k < 4; k++) {
      bc_in[2 * k + 0] = bc_in_min_(ib, k);
      bc_in[2 * k + 1] = bc_in_max_(ib, k);
      orcheck[k] = (bc_in[2 * k] != VUNDEF) || (bc_in[2 * k + 1] != VUNDEF);
      orcsum += orcheck[k];
      bitconf |= 1 << k;
    }

    int sign1[4], sign2[4], sign3[4], sign4[4];
    for(int k1 = 0; k1 < 8; k1++) {
      if(bc_in[k1] != VUNDEF) {
        int vdx[4], vex[4];
        for(int ii = 0; ii < 4; ii++) {
          sign1[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];
          if(ii == k1/2)
            sign1[ii] = bc_sign_[k1], vdx[ii] = bc_in[k1], vex[ii] = bc_in[k1];
        }

        for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
          for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
            for(int jy = vdx[1]; jy <= vex[1]; jy++) {
              for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                int rx  = (nx_  + jx  - sign1[0]) % nx_;
                int ry  = (ny_  + jy  - sign1[1]) % ny_;
                int rvx = (nvx_ + jvx - sign1[2]) % nvx_;
                int rvy = (nvy_ + jvy - sign1[3]) % nvy_;
                float64 fsum = 0.;
                float64 alphap1 = alpha_;
                for(int j1 = 1; j1 <= MMAX; j1++) {
                  fsum += halo_fn_(rx  + sign1[0] * j1 - local_xstart_  + HALO_PTS, 
                                   ry  + sign1[1] * j1 - local_ystart_  + HALO_PTS,
                                   rvx + sign1[2] * j1 - local_vxstart_ + HALO_PTS,
                                   rvy + sign1[3] * j1 - local_vystart_ + HALO_PTS) * alphap1;
                  alphap1 *= alpha_;
                }
                int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                              jy  - halo_min[1],
                                              jvx - halo_min[2], 
                                              jvy - halo_min[3], 
                                              halo_nx, halo_ny, halo_nvx, halo_nvy);
                buf_(idx, ib) = fsum;
              } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
            } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
          } // for(int jvx = vdx[2]; jvx <= vex[2]; jvx++)
        } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
      } // if(bc_in[k1] != VUNDEF)
    } // for(int k1 = 0; k1 < 8; k1++)

    if(orcsum > 1) {
      for(int k1 = 0; k1 < 8; k1++) {
        for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++) {
          if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF) {
            int vdx[4], vex[4];
            for(int ii = 0; ii < 4; ii++) {
              sign1[ii] = 0, sign2[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];

              if(ii == k1/2)
                sign1[ii] = bc_sign_[k1], vex[ii] = vdx[ii] = bc_in[k1];
              if(ii == k2/2)
                sign2[ii] = bc_sign_[k2], vex[ii] = vdx[ii] = bc_in[k2];
            }

            for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
              for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                  for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                    int rx  = (nx_  + jx  - sign1[0] - sign2[0]) % nx_;
                    int ry  = (ny_  + jy  - sign1[1] - sign2[1]) % ny_;
                    int rvx = (nvx_ + jvx - sign1[2] - sign2[2]) % nvx_;
                    int rvy = (nvy_ + jvy - sign1[3] - sign2[3]) % nvy_;

                    float64 fsum = 0.;
                    float64 alphap2 = alpha_;
                    for(int j2 = 1; j2 <= MMAX; j2++) {
                      float64 alphap1 = alpha_ * alphap2;  
                      for(int j1 = 1; j1 <= MMAX; j1++) {
                        fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2 - local_xstart_  + HALO_PTS, 
                                         ry  + sign1[1] * j1 + sign2[1] * j2 - local_ystart_  + HALO_PTS,
                                         rvx + sign1[2] * j1 + sign2[2] * j2 - local_vxstart_ + HALO_PTS,
                                         rvy + sign1[3] * j1 + sign2[3] * j2 - local_vystart_ + HALO_PTS) * alphap1;
                        alphap1 *= alpha_;
                      }
                      alphap2 *= alpha_;
                    }
                    int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                  jy  - halo_min[1],
                                                  jvx - halo_min[2],
                                                  jvy - halo_min[3],
                                                  halo_nx, halo_ny, halo_nvx, halo_nvy);
                    buf_(idx, ib) = fsum;
                  } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
                } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
              } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
            } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
          } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
        } // for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++)
      } // for(int k1 = 0; k1 < 8; k1++)
    } // if(orcsum > 1) {

    if(orcsum > 2) {
      for(int k1 = 0; k1 < 8; k1++) {
        for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++) {
          for(int k3 = 2 * (1 + k2/2); k3 < 8; k3++) {
            if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF) {
              int vdx[4], vex[4];
              for(int ii = 0; ii < 4; ii++) {
                sign1[ii] = 0, sign2[ii] = 0, sign3[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];

                if(ii == k1/2)
                  sign1[ii] = bc_sign_[k1], vex[ii] = vdx[ii] = bc_in[k1];
                if(ii == k2/2)
                  sign2[ii] = bc_sign_[k2], vex[ii] = vdx[ii] = bc_in[k2];
                if(ii == k3/2)
                  sign3[ii] = bc_sign_[k3], vex[ii] = vdx[ii] = bc_in[k3];
              }

              for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                  for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                    for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                      int rx  = (nx_  + jx  - sign1[0] - sign2[0] - sign3[0]) % nx_;
                      int ry  = (ny_  + jy  - sign1[1] - sign2[1] - sign3[1]) % ny_;
                      int rvx = (nvx_ + jvx - sign1[2] - sign2[2] - sign3[2]) % nvx_;
                      int rvy = (nvy_ + jvy - sign1[3] - sign2[3] - sign3[3]) % nvy_;
                      float64 fsum = 0.;
                      float64 alphap3 = alpha_;
                      for(int j3 = 1; j3 <= MMAX; j3++) {
                        float64 alphap2 = alpha_ * alphap3;
                        for(int j2 = 1; j2 <= MMAX; j2++) {
                          float64 alphap1 = alpha_ * alphap2;
                          for(int j1 = 1; j1 <= MMAX; j1++) {
                            fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2 + sign3[0] * j3 - local_xstart_  + HALO_PTS, 
                                             ry  + sign1[1] * j1 + sign2[1] * j2 + sign3[1] * j3 - local_ystart_  + HALO_PTS,
                                             rvx + sign1[2] * j1 + sign2[2] * j2 + sign3[2] * j3 - local_vxstart_ + HALO_PTS,
                                             rvy + sign1[3] * j1 + sign2[3] * j2 + sign3[3] * j3 - local_vystart_ + HALO_PTS) * alphap1;
                            alphap1 *= alpha_;
                          }
                          alphap2 *= alpha_;
                        }
                        alphap3 *= alpha_;
                      }
                      int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                    jy  - halo_min[1],
                                                    jvx - halo_min[2],
                                                    jvy - halo_min[3],
                                                    halo_nx, halo_ny, halo_nvx, halo_nvy);
                      buf_(idx, ib) = fsum;
                    } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
                  } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
                } // for(int jvx = vdx[2]; jvx <= vex[2]; jvx++)
              } // for(int jvy = vdx[3]; jvy <= vex[3]; jvy++)
            } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
          } // for(int k3 = 2 * (1 + k2/2); k3 < 8; k3++)
        } // for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++)
      } // for(int k1 = 0; k1 < 8; k1++)
    } // if(orcsum > 2) {

    if(orcsum > 3) {
      for(int k1 = 0; k1 < 2; k1++) {
        for(int k2 = 2; k2 < 4; k2++) {
          for(int k3 = 4; k3 < 6; k3++) {
            for(int k4 = 6; k4 < 8; k4++) {
              if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF && bc_in[k4] != VUNDEF) {
                int vdx[4], vex[4];
                for(int ii = 0; ii < 4; ii++) {
                  sign1[ii] = 0, sign2[ii] = 0, sign3[ii] = 0, sign4[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];

                  if(ii == k1/2)
                    sign1[ii] = bc_sign_[k1], vex[ii] = vdx[ii] = bc_in[k1];
                  if(ii == k2/2)
                    sign2[ii] = bc_sign_[k2], vex[ii] = vdx[ii] = bc_in[k2];
                  if(ii == k3/2)
                    sign3[ii] = bc_sign_[k3], vex[ii] = vdx[ii] = bc_in[k3];
                  if(ii == k4/2)
                    sign4[ii] = bc_sign_[k4], vex[ii] = vdx[ii] = bc_in[k4];
                }

                for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                  for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                    for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                      for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                        int rx  = (nx_  + jx  - sign1[0]) % nx_;
                        int ry  = (ny_  + jy  - sign2[1]) % ny_;
                        int rvx = (nvx_ + jvx - sign3[2]) % nvx_;
                        int rvy = (nvy_ + jvy - sign4[3]) % nvy_;

                        float64 fsum = 0.;
                        float64 alphap4 = alpha_;
                        for(int j4 = 1; j4 <= MMAX; j4++) {
                          float64 alphap3 = alpha_ * alphap4;
                          for(int j3 = 1; j3 <= MMAX; j3++) {
                            float64 alphap2 = alpha_ * alphap3;
                            for(int j2 = 1; j2 <= MMAX; j2++) {
                              float64 alphap1 = alpha_ * alphap2;
                              for(int j1 = 1; j1 <= MMAX; j1++) {
                                fsum += halo_fn_(rx  + sign1[0] * j1 - local_xstart_  + HALO_PTS, 
                                                 ry  + sign2[1] * j2 - local_ystart_  + HALO_PTS,
                                                 rvx + sign3[2] * j3 - local_vxstart_ + HALO_PTS,
                                                 rvy + sign4[3] * j4 - local_vystart_ + HALO_PTS) * alphap1;
                                alphap1 *= alpha_;
                              }
                              alphap2 *= alpha_;
                            }
                            alphap3 *= alpha_;
                          }
                          alphap4 *= alpha_;
                        }
                        int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                      jy  - halo_min[1],
                                                      jvx - halo_min[2],
                                                      jvy - halo_min[3],
                                                      halo_nx, halo_ny, halo_nvx, halo_nvy);
                        buf_(idx, ib) = fsum;
                      } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
                    } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
                  } // for(int jvx = vdx[2]; jvx <= vex[2]; jvx++)
                } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
              } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
            } // for(int k4 = 6; k4 < 8; k4++)
          } // for(int k3 = 4; k3 < 6; k3++)
        } // for(int k2 = 2; k2 < 4; k2++)
      } // for(int k1 = 0; k1 < 8; k1++)
    } // if(orcsum > 3)
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ix, const int iy, const int ivx, const int ib) const {
    int halo_min[4], halo_max[4];
    for(int k = 0; k < DIMENSION; k++) {
      halo_min[k] = xmin_(ib, k);
      halo_max[k] = xmax_(ib, k);
    }

    const int halo_nx  = halo_max[0] - halo_min[0] + 1;
    const int halo_ny  = halo_max[1] - halo_min[1] + 1;
    const int halo_nvx = halo_max[2] - halo_min[2] + 1;
    const int halo_nvy = halo_max[3] - halo_min[3] + 1;

    int bc_in[8], orcheck[4];

    int orcsum = 0;
    char bitconf = 0;
    for(int k = 0; k < 4; k++) {
      bc_in[2 * k + 0] = bc_in_min_(ib, k);
      bc_in[2 * k + 1] = bc_in_max_(ib, k);
      orcheck[k] = (bc_in[2 * k] != VUNDEF) || (bc_in[2 * k + 1] != VUNDEF);
      orcsum += orcheck[k];
      bitconf |= 1 << k;
    }

    int sign1[4], sign2[4], sign3[4], sign4[4];
    for(int k1 = 0; k1 < 8; k1++) {
      if(bc_in[k1] != VUNDEF) {
        int vdx[4], vex[4];
        for(int ii = 0; ii < 4; ii++) {
          sign1[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];
          if(ii == k1/2) {
            sign1[ii] = bc_sign_[k1], vdx[ii] = bc_in[k1], vex[ii] = bc_in[k1];
          }
        }

        const int jx  = ix  + vdx[0];
        const int jy  = iy  + vdx[1];
        const int jvx = ivx + vdx[2];
        if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
          for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
            int rx  = (nx_  + jx  - sign1[0]) % nx_;
            int ry  = (ny_  + jy  - sign1[1]) % ny_;
            int rvx = (nvx_ + jvx - sign1[2]) % nvx_;
            int rvy = (nvy_ + jvy - sign1[3]) % nvy_;
            float64 fsum = 0.;
            float64 alphap1 = alpha_;
            for(int j1 = 1; j1 <= MMAX; j1++) {
              fsum += halo_fn_(rx  + sign1[0] * j1 - local_xstart_  + HALO_PTS, 
                               ry  + sign1[1] * j1 - local_ystart_  + HALO_PTS,
                               rvx + sign1[2] * j1 - local_vxstart_ + HALO_PTS,
                               rvy + sign1[3] * j1 - local_vystart_ + HALO_PTS) * alphap1;
              alphap1 *= alpha_;
            }
            int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                          jy  - halo_min[1],
                                          jvx - halo_min[2], 
                                          jvy - halo_min[3], 
                                          halo_nx, halo_ny, halo_nvx, halo_nvy);
            buf_(idx, ib) = fsum;
          } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
        } // if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] )
      } // if(bc_in[k1] != VUNDEF)
    } // for(int k1 = 0; k1 < 8; k1++)

    if(orcsum > 1) {
      for(int k1 = 0; k1 < 8; k1++) {
        for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++) {
          if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF) {
            int vdx[4], vex[4], sign1[4], sign2[4];
            for(int ii = 0; ii < 4; ii++) {
              sign1[ii] = 0, sign2[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];

              if(ii == k1/2)
                sign1[ii] = bc_sign_[k1], vex[ii] = vdx[ii] = bc_in[k1];
              if(ii == k2/2)
                sign2[ii] = bc_sign_[k2], vex[ii] = vdx[ii] = bc_in[k2];
            }

            const int jx  = ix  + vdx[0];
            const int jy  = iy  + vdx[1];
            const int jvx = ivx + vdx[2];
            if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
              int rx  = (nx_  + jx  - sign1[0] - sign2[0]) % nx_;
              int ry  = (ny_  + jy  - sign1[1] - sign2[1]) % ny_;
              int rvx = (nvx_ + jvx - sign1[2] - sign2[2]) % nvx_;
              for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                int rvy = (nvy_ + jvy - sign1[3] - sign2[3]) % nvy_;
                float64 fsum = 0.;
                float64 alphap2 = alpha_;
                for(int j2 = 1; j2 <= MMAX; j2++) {
                  float64 alphap1 = alpha_ * alphap2;  
                  for(int j1 = 1; j1 <= MMAX; j1++) {
                    fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2 - local_xstart_  + HALO_PTS, 
                                     ry  + sign1[1] * j1 + sign2[1] * j2 - local_ystart_  + HALO_PTS,
                                     rvx + sign1[2] * j1 + sign2[2] * j2 - local_vxstart_ + HALO_PTS,
                                     rvy + sign1[3] * j1 + sign2[3] * j2 - local_vystart_ + HALO_PTS) * alphap1;
                    alphap1 *= alpha_;
                  }
                  alphap2 *= alpha_;
                }
                int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                              jy  - halo_min[1],
                                              jvx - halo_min[2],
                                              jvy - halo_min[3],
                                              halo_nx, halo_ny, halo_nvx, halo_nvy);
                buf_(idx, ib) = fsum;
              } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
            } // if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] )
          } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
        } // for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++)
      } // for(int k1 = 0; k1 < 8; k1++)
    } // if(orcsum > 1)

    if(orcsum > 2) {
      for(int k1 = 0; k1 < 8; k1++) {
        for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++) {
          for(int k3 = 2 * (1 + k2/2); k3 < 8; k3++) {
            if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF) {
              int vdx[4], vex[4];
              for(int ii = 0; ii < 4; ii++) {
                sign1[ii] = 0, sign2[ii] = 0, sign3[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];

                if(ii == k1/2)
                  sign1[ii] = bc_sign_[k1], vex[ii] = vdx[ii] = bc_in[k1];
                if(ii == k2/2)
                  sign2[ii] = bc_sign_[k2], vex[ii] = vdx[ii] = bc_in[k2];
                if(ii == k3/2)
                  sign3[ii] = bc_sign_[k3], vex[ii] = vdx[ii] = bc_in[k3];
              }

              const int jx  = ix  + vdx[0];
              const int jy  = iy  + vdx[1];
              const int jvx = ivx + vdx[2];
              if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
                int rx  = (nx_  + jx  - sign1[0] - sign2[0] - sign3[0]) % nx_;
                int ry  = (ny_  + jy  - sign1[1] - sign2[1] - sign3[1]) % ny_;
                int rvx = (nvx_ + jvx - sign1[2] - sign2[2] - sign3[2]) % nvx_;
                for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                  int rvy = (nvy_ + jvy - sign1[3] - sign2[3] - sign3[3]) % nvy_;
                  float64 fsum = 0.;
                  float64 alphap3 = alpha_;
                  for(int j3 = 1; j3 <= MMAX; j3++) {
                    float64 alphap2 = alpha_ * alphap3;
                    for(int j2 = 1; j2 <= MMAX; j2++) {
                      float64 alphap1 = alpha_ * alphap2;
                      for(int j1 = 1; j1 <= MMAX; j1++) {
                        fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2 + sign3[0] * j3 - local_xstart_  + HALO_PTS, 
                                         ry  + sign1[1] * j1 + sign2[1] * j2 + sign3[1] * j3 - local_ystart_  + HALO_PTS,
                                         rvx + sign1[2] * j1 + sign2[2] * j2 + sign3[2] * j3 - local_vxstart_ + HALO_PTS,
                                         rvy + sign1[3] * j1 + sign2[3] * j2 + sign3[3] * j3 - local_vystart_ + HALO_PTS) * alphap1;
                        alphap1 *= alpha_;
                      }
                      alphap2 *= alpha_;
                    }
                    alphap3 *= alpha_;
                  }
                  int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                jy  - halo_min[1],
                                                jvx - halo_min[2],
                                                jvy - halo_min[3],
                                                halo_nx, halo_ny, halo_nvx, halo_nvy);
                  buf_(idx, ib) = fsum;
                } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
              } // if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] )
            } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
          } // for(int k3 = 2 * (1 + k2/2); k3 < 8; k3++)
        } // for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++)
      } // for(int k1 = 0; k1 < 8; k1++)
    } // if(orcsum > 2)

    if(orcsum > 3) {
      for(int k1 = 0; k1 < 2; k1++) {
        for(int k2 = 2; k2 < 4; k2++) {
          for(int k3 = 4; k3 < 6; k3++) {
            for(int k4 = 6; k4 < 8; k4++) {
              if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF && bc_in[k4] != VUNDEF) {
                int vdx[4], vex[4];
                for(int ii = 0; ii < 4; ii++) {
                  sign1[ii] = 0, sign2[ii] = 0, sign3[ii] = 0, sign4[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];

                  if(ii == k1/2)
                    sign1[ii] = bc_sign_[k1], vex[ii] = vdx[ii] = bc_in[k1];
                  if(ii == k2/2)
                    sign2[ii] = bc_sign_[k2], vex[ii] = vdx[ii] = bc_in[k2];
                  if(ii == k3/2)
                    sign3[ii] = bc_sign_[k3], vex[ii] = vdx[ii] = bc_in[k3];
                  if(ii == k4/2)
                    sign4[ii] = bc_sign_[k4], vex[ii] = vdx[ii] = bc_in[k4];
                }

                const int jx  = ix  + vdx[0];
                const int jy  = iy  + vdx[1];
                const int jvx = ivx + vdx[2];
                if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
                  int rx  = (nx_  + jx  - sign1[0]) % nx_;
                  int ry  = (ny_  + jy  - sign2[1]) % ny_;
                  int rvx = (nvx_ + jvx - sign3[2]) % nvx_;
                  for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                    int rvy = (nvy_ + jvy - sign4[3]) % nvy_;
                    float64 fsum = 0.;
                    float64 alphap4 = alpha_;
                    for(int j4 = 1; j4 <= MMAX; j4++) {
                      float64 alphap3 = alpha_ * alphap4;
                      for(int j3 = 1; j3 <= MMAX; j3++) {
                        float64 alphap2 = alpha_ * alphap3;
                        for(int j2 = 1; j2 <= MMAX; j2++) {
                          float64 alphap1 = alpha_ * alphap2;
                          for(int j1 = 1; j1 <= MMAX; j1++) {
                            fsum += halo_fn_(rx  + sign1[0] * j1 - local_xstart_  + HALO_PTS, 
                                             ry  + sign2[1] * j2 - local_ystart_  + HALO_PTS,
                                             rvx + sign3[2] * j3 - local_vxstart_ + HALO_PTS,
                                             rvy + sign4[3] * j4 - local_vystart_ + HALO_PTS) * alphap1;
                            alphap1 *= alpha_;
                          }
                          alphap2 *= alpha_;
                        }
                        alphap3 *= alpha_;
                      }
                      alphap4 *= alpha_;
                    }
                    int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                  jy  - halo_min[1],
                                                  jvx - halo_min[2],
                                                  jvy - halo_min[3],
                                                  halo_nx, halo_ny, halo_nvx, halo_nvy);
                    buf_(idx, ib) = fsum;
                  } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
                } // if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] )
              } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
            } // for(int k4 = 6; k4 < 8; k4++)
          } // for(int k3 = 4; k3 < 6; k3++)
        } // for(int k2 = 2; k2 < 4; k2++)
      } // for(int k1 = 0; k1 < 8; k1++)
    } // if(orcsum > 3)
  }
};

struct boundary_condition_orig {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  Config         *conf_;
  RealView4D     halo_fn_;
  RealLeftView2D buf_;
  Halos          send_halos_;
  RangeView2D    xmin_, xmax_;
  RangeView2D    lxmin_, lxmax_;

  float64 alpha_;
  // Global domain size
  int nx_, ny_, nvx_, nvy_;

  // Local domain min and max
  int local_xstart_, local_ystart_, local_vxstart_, local_vystart_;

  boundary_condition_orig(Config *conf, RealView4D halo_fn, Halos send_halos)
    : conf_(conf), halo_fn_(halo_fn), send_halos_(send_halos) {
    buf_  = send_halos_.buf_;
    xmin_ = send_halos_.xmin_;
    xmax_ = send_halos_.xmax_;
    lxmin_ = send_halos_.lxmin_;
    lxmax_ = send_halos_.lxmax_;
    const Domain *dom = &(conf->dom_);
    nx_  = dom->nxmax_[0];
    ny_  = dom->nxmax_[1];
    nvx_ = dom->nxmax_[2];
    nvy_ = dom->nxmax_[3];
    alpha_ = sqrt(3) - 2;

    // Without halo region
    local_xstart_  = dom->local_nxmin_[0];
    local_ystart_  = dom->local_nxmin_[1];
    local_vxstart_ = dom->local_nxmin_[2];
    local_vystart_ = dom->local_nxmin_[3];
  }

  /* @ packed_values[0,1,2,3]: check, mini, maxi, r
   * @ min: si = halo->lxmin_ - 3
   * @ max: ei = halo->lxmax_ + 3
   * @ n:   dg = dom.nxmax_
   */
  KOKKOS_INLINE_FUNCTION
  void check(int *packed_values, const int idx, const int min, const int max, const int n) const {
    int check, mini, maxi, r;
    if(idx == min) {
      check = -1;
      mini  = 1;
      maxi  = MMAX;
      r     = (n + idx + 1) % n;
    } else if(idx == max) {
      check = -1;
      mini  = 1;
      maxi  = MMAX;
      r     = (n + idx - 1) % n;
    } else {
      check = 0;
      mini  = 0;
      maxi  = 0;
      r     = (n + idx) % n;
    }
    packed_values[0] = check;
    packed_values[1] = mini;
    packed_values[2] = maxi;
    packed_values[3] = r;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int ix, const int iy, const int ivx, const int ib) const {
    const int ix_min  = xmin_(ib, 0), ix_max  = xmin_(ib, 0); 
    const int iy_min  = xmin_(ib, 1), iy_max  = xmax_(ib, 1);
    const int ivx_min = xmin_(ib, 2), ivx_max = xmax_(ib, 2);
    const int ivy_min = xmin_(ib, 3), ivy_max = xmax_(ib, 3);
    const int local_xmin  = lxmin_(ib, 0), local_xmax  = lxmax_(ib, 0);
    const int local_ymin  = lxmin_(ib, 1), local_ymax  = lxmax_(ib, 1);
    const int local_vxmin = lxmin_(ib, 2), local_vxmax = lxmax_(ib, 2);
    const int local_vymin = lxmin_(ib, 3), local_vymax = lxmax_(ib, 3);

    //const int halo_nx  = ix_max  - ix_min  + 1;
    //const int halo_ny  = iy_max  - iy_min  + 1;
    //const int halo_nvx = ivx_max - ivx_min + 1;
    const int halo_nvy = ivy_max - ivy_min + 1;

    const int jx  = ix  + ix_min;
    const int jy  = iy  + iy_min;
    const int jvx = ivx + ivx_min;
    int packed_values_x[4];
    int packed_values_y[4];
    int packed_values_vx[4];
    int packed_values_vy[4];

    if ( (jx <= ix_max) && (jy <= iy_max) && (jvx <= ivx_max) ) {
      check(packed_values_x,  ix,  local_xmin,  local_xmax, nx_);
      check(packed_values_y,  iy,  local_ymin,  local_ymax, ny_);
      check(packed_values_vx, ivx, local_vxmin, local_vxmax, nvx_);
      int check_x  = packed_values_x[0],  mini_x  = packed_values_x[1],  maxi_x  = packed_values_x[2],  rx  = packed_values_x[3];
      int check_y  = packed_values_y[0],  mini_y  = packed_values_y[1],  maxi_y  = packed_values_y[2],  ry  = packed_values_y[3];
      int check_vx = packed_values_vx[0], mini_vx = packed_values_vx[1], maxi_vx = packed_values_vx[2], rvx = packed_values_vx[3];
      for(int ivy = 0; ivy < halo_nvy; ivy++) {
        check(packed_values_vy, ivy, local_vymin, local_vymax, nvy_);
        int check_vy = packed_values_vy[0], mini_vy = packed_values_vy[1], maxi_vy = packed_values_vy[2], rvy = packed_values_vy[3];

        if(check_x != 0 || check_y != 0 || check_vx != 0 || check_vy != 0) {
          float64 fsum = 0.;
          for(int j3=mini_vy; j3<maxi_vy; j3++) {
            int jvy = rvy + check_vy * j3 + HALO_PTS;
            for(int j2=mini_vx; j2<maxi_vx; j2++) {
              int jvx = rvx + check_vx * j2 + HALO_PTS;
              for(int j1=mini_y; j1<maxi_y; j1++) {
                int jy = ry + check_y * j1 + HALO_PTS;
                for(int j0=mini_x; j0<maxi_x; j0++) {
                  float64 alphapow = pow(alpha_, j0 + j1 + j2 + j3);
                  /*
                   * assert(sn[0] <= jx && jx <= en[0]);
                   * assert(sn[1] <= jy && jy <= en[1]);
                   * assert(sn[2] <= jvx && jvx <= en[2]);
                   * assert(sn[3] <= jvy && jvy <= en[3]);
                   */
                  int jx = rx + check_x * j0 + HALO_PTS;
                  fsum += halo_fn_(jx, jy, jvx, jvy) * alphapow;
                }
              }
            }
          }
          #ifdef ORIGALGO
            int idx = Index::coord_4D2int(ix, iy, ivx, ivy, halo_nx, halo_ny, halo_nvx, halo_nvy);
            buf_(idx, ib) = fsum;
         /*
          #else
          */
          #endif
        } // if(check_x != 0 || check_y != 0 || check_vx != 0 || check_vy != 0)
      }
    }
  }
};

#endif
