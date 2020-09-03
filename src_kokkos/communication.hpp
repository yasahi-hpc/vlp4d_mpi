#ifndef __COMM_H__
#define __COMM_H__

#include <vector>
#include <mpi.h>
#include <algorithm>
#include <cassert>
#include "types.h"
#include "Config.hpp"
#include "Index.hpp"
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
  typedef Kokkos::View<int*[DIMENSION][DIMENSION], execution_space> RangeView3D;
  RealView2D buf_;
  RealView1D buf_flatten_;
  RangeView2D xmin_;
  RangeView2D xmax_;
  RangeView2D bc_in_min_;
  RangeView2D bc_in_max_;
  RangeView2D lxmin_;
  RangeView2D lxmax_;

  shape_t<DIMENSION> nhalo_max_;
  int size_;     // buffer size of each halo
  int nb_halos_; // the number of halos
  int pid_, nbp_;

  std::vector<int> sizes_;
  std::vector<int> pids_;
  std::vector<int> tags_;

  /* Used for merge */
  RangeView2D map_;         // f -> flatten_buf
  IntView2D   flatten_map_; // buf -> flatten_buf
  int offset_local_copy_; // The head address for the local copy
  std::vector<int> merged_sizes_; // buffer size of each halo
  std::vector<int> merged_pids_; // merged process id to interact with
  std::vector<int> merged_tags_; //
  std::vector<int> id_map_; // mapping the original id to the merged id
  std::vector<int> merged_heads_;
  int total_size_; // total size of all the buffers
  int nb_merged_halos_; // number of halos after merge
  int nb_nocomms_;

  /* Return the */
  float64* head(const int i) {
    float64* dptr_buf = buf_flatten_.data() + merged_heads_.at(i);
    return dptr_buf;
  }

  int merged_size(const int i) {
    return merged_sizes_.at(i);
  }
  
  int merged_pid(const int i) {
    return merged_pids_.at(i);
  }
   
  int merged_tag(const int i) {
    return merged_tags_.at(i);
  }
   
  int nb_reqs(){ return (nbp_ - nb_nocomms_); };

  void mergeLists(Config *conf, const std::string name) {
    nb_nocomms_ = 1;
    int total_size = 0;
    std::vector<int> id_map( nb_halos_ );
    std::vector< std::vector<int> > group_same_dst;

    for(int pid = 0; pid < nbp_; pid++) {
      if(pid == pid_) {
        offset_local_copy_ = total_size;
      }
      std::vector<int> same_dst;
      for(auto it = pids_.begin(); it != pids_.end(); it++) {
        if(*it == pid) {
          int dst_id = std::distance(pids_.begin(), it);
          same_dst.push_back( std::distance(pids_.begin(), it) );
        }
      }
                         
      // Save merged data for current pid
      int size = 0;
      int tag = 0;
      if(same_dst.empty()) {
        if(pid != pid_) nb_nocomms_++;
      } else {
        tag = tags_[same_dst[0]]; // Use the tag of first element
      }
      for(auto it: same_dst) {
        id_map.at(it) = total_size + size;
        size += sizes_[it];
      }
       
      merged_sizes_.push_back(size);
      merged_pids_.push_back(pid);
      merged_tags_.push_back(tag);
      total_size += size; // Size of the total size summed up for previous pids
      group_same_dst.push_back(same_dst);
    }

    total_size_  = total_size;
    map_         = RangeView2D("map", total_size); // This is used for receive buffer
    buf_flatten_ = RealView1D(name + "_buf_flat", total_size);
    flatten_map_ = IntView2D("flatten_map", total_size, 2); // storing (idx_in_buf, buf_id)

    typename RangeView2D::HostMirror h_map  = Kokkos::create_mirror_view(map_);
    typename RangeView2D::HostMirror h_xmin = Kokkos::create_mirror_view(xmin_);
    typename RangeView2D::HostMirror h_xmax = Kokkos::create_mirror_view(xmax_);
    Kokkos::deep_copy(h_xmin, xmin_);
    Kokkos::deep_copy(h_xmax, xmax_);

    const Domain *dom = &(conf->dom_);
    int nx_max  = dom->nxmax_[0];
    int ny_max  = dom->nxmax_[1];
    int nvx_max = dom->nxmax_[2];
    int nvy_max = dom->nxmax_[3];
     
    int local_xstart  = dom->local_nxmin_[0];
    int local_ystart  = dom->local_nxmin_[1];
    int local_vxstart = dom->local_nxmin_[2];
    int local_vystart = dom->local_nxmin_[3];

    int idx_flatten = 0;
    typename View2D<int>::HostMirror h_flatten_map = Kokkos::create_mirror_view(flatten_map_);
    for(auto same_dst: group_same_dst) {
      // Keeping the head index of each halo sets for MPI communication
      merged_heads_.push_back(idx_flatten);
      for(auto it: same_dst) {
        const int ix_min  = h_xmin(it, 0), ix_max  = h_xmax(it, 0);
        const int iy_min  = h_xmin(it, 1), iy_max  = h_xmax(it, 1);
        const int ivx_min = h_xmin(it, 2), ivx_max = h_xmax(it, 2);
        const int ivy_min = h_xmin(it, 3), ivy_max = h_xmax(it, 3);
        
        const int nx  = ix_max  - ix_min  + 1;
        const int ny  = iy_max  - iy_min  + 1;
        const int nvx = ivx_max - ivx_min + 1;
        const int nvy = ivy_max - ivy_min + 1;
        
        for(int ivy = ivy_min; ivy <= ivy_max; ivy++) {
          for(int ivx = ivx_min; ivx <= ivx_max; ivx++) {
            for(int iy = iy_min; iy <= iy_max; iy++) {
              for(int ix = ix_min; ix <= ix_max; ix++) {
                int idx = Index::coord_4D2int(ix-ix_min,
                                              iy-iy_min,
                                              ivx-ivx_min,
                                              ivy-ivy_min,
                                              nx, ny, nvx, nvy);
              
                // h_map is used for receive buffer
                h_map(idx_flatten, 0) = ix;
                h_map(idx_flatten, 1) = iy;
                h_map(idx_flatten, 2) = ivx;
                h_map(idx_flatten, 3) = ivy;
                
                // h_flatten_map is used for send buffer
                h_flatten_map(idx_flatten, 0) = idx;
                h_flatten_map(idx_flatten, 1) = it;
                idx_flatten++;
              }
            }
          }
        }
      }
    }
    Kokkos::deep_copy(flatten_map_, h_flatten_map);
    Kokkos::deep_copy(map_, h_map);
  }

  void set(Config *conf, std::vector<Halo> &list, const std::string name, const int nb_process, const int pid) {
    nbp_ = nb_process;
    pid_ = pid;
    nb_merged_halos_ = nbp_;

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

    std::vector<int> nx_halos, ny_halos, nvx_halos, nvy_halos;
    for(size_t i = 0; i < nb_halos_; i++) {
      Halo *halo = &(list[i]);
      int tmp_size = (halo->xmax_[0] - halo->xmin_[0] + 1) * (halo->xmax_[1] - halo->xmin_[1] + 1)
                   * (halo->xmax_[2] - halo->xmin_[2] + 1) * (halo->xmax_[3] - halo->xmin_[3] + 1);
      sizes_.push_back(tmp_size);
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
    auto max_size = std::max_element(sizes_.begin(), sizes_.end());
    size_ = *max_size;

    nhalo_max_[0] = *std::max_element(nx_halos.begin(),  nx_halos.end());
    nhalo_max_[1] = *std::max_element(ny_halos.begin(),  ny_halos.end());
    nhalo_max_[2] = *std::max_element(nvx_halos.begin(), nvx_halos.end());
    nhalo_max_[3] = *std::max_element(nvy_halos.begin(), nvy_halos.end());

    buf_ = RealView2D(name + "_buffer", size_, nb_halos_);
    mergeLists(conf, name);
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
  Halos *recv_buffers_;

  // List of halo buffers (sending side)
  std::vector<Halo> send_list_;
  Halos *send_buffers_;

  // The local box for the very local MPI domain
  Urbnode *node_;

  // Use spline or not
  bool spline_;

  // Global domain size
  int nxmax_[DIMENSION];

public:
  Distrib(int &nargs, char **argv) : spline_(true), recv_buffers_(nullptr), send_buffers_(nullptr) {
    int required = MPI_THREAD_SERIALIZED;
    int provided;

    // Initialize MPI
    MPI_Init_thread(&nargs, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp_);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid_);
  };

  ~Distrib(){};
  void cleanup() {
    if(recv_buffers_ != nullptr) delete recv_buffers_;
    if(send_buffers_ != nullptr) delete send_buffers_;
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
  void neighboursList(Config *conf, RealOffsetView4D halo_fn); 
  void bookHalo(Config *conf);

  // Communication
  void exchangeHalo(Config *conf, RealOffsetView4D halo_fn, std::vector<Timer*> &timers);
private:

  template <class ViewType4D>
  void getNeighbours(const Config *conf, const ViewType4D &halo_fn, int xrange[8],
                     std::vector<Halo> &hlist, int lxmin[4], int lxmax[4], int count) {
    std::vector<Halo> vhalo;
    uint8 neighbours[nbp_];
    uint32 nb_neib = 0;
     
    for(uint32 j=0; j<nbp_; j++)
      neighbours[j] = 255;
     
    vhalo.clear();
    for(int ivy = xrange[6]; ivy <= xrange[7]; ivy++) {
      for(int ivx = xrange[4]; ivx <= xrange[5]; ivx++) {
        for(int iy = xrange[2]; iy <= xrange[3]; iy++) {
          for(int ix = xrange[0]; ix <= xrange[1]; ix++) {
            const uint32 neibid = round(halo_fn(ix, iy, ivx, ivy));

            if(neighbours[neibid] == 255) {
              Halo myneib;
              neighbours[neibid] = nb_neib;
              myneib.pid_     = neibid;
              myneib.xmin_[0] = ix;
              myneib.xmin_[1] = iy;
              myneib.xmin_[2] = ivx;
              myneib.xmin_[3] = ivy;
              myneib.xmax_[0] = ix;
              myneib.xmax_[1] = iy;
              myneib.xmax_[2] = ivx;
              myneib.xmax_[3] = ivy;
              myneib.tag_ = count;
              for(int k = 0; k < 4; k++) {
                myneib.lxmin_[k] = lxmin[k];
                myneib.lxmax_[k] = lxmax[k];
              }
              vhalo.push_back(myneib);
              nb_neib++;
            }//if(neighbours[neibid] == 255)
            uint8 neighbour = neighbours[neibid];
            vhalo[neighbour].xmax_[0] = ix;
            vhalo[neighbour].xmax_[1] = iy;
            vhalo[neighbour].xmax_[2] = ivx;
            vhalo[neighbour].xmax_[3] = ivy;
          }//for(int32 ix = xrange[0]; ix <= xrange[1]; ix++)
        }//for(int32 iy = xrange[2]; iy <= xrange[3]; iy++)
      }//for(int32 ivx = xrange[4]; ivx <= xrange[5]; ivx++)
    }//for(int32 ivy = xrange[6]; ivy <= xrange[7]; ivy++)

    hlist.insert(hlist.end(), vhalo.begin(), vhalo.end());
  }

  void packAndBoundary(Config *conf, RealOffsetView4D halo_fn);
  void unpack(RealOffsetView4D halo_fn);

  int mergeElts(std::vector<Halo> &v, std::vector<Halo>::iterator &f, std::vector<Halo>::iterator &g);

  // Wrapper for MPI communication
  void Isend(int &creq, std::vector<MPI_Request> &req);
  
  // Wrapper for MPI communication
  void Irecv(int &creq, std::vector<MPI_Request> &req);
  
  // Wrapper for MPI communication
  void Waitall(const int creq, std::vector<MPI_Request> &req, std::vector<MPI_Status> &stat) {
    const int nbreq = req.size();
    assert(creq == nbreq);
    MPI_Waitall(nbreq, req.data(), stat.data());
  };

private:

  DISALLOW_DEFAULT(Distrib);
  DISALLOW_COPY_AND_ASSIGN(Distrib);
};

// functors
struct pack {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  Config         *conf_;
  RealOffsetView4D halo_fn_;
  RealView2D buf_;
  Halos          *send_halos_;
  RangeView2D    xmin_, xmax_;
  int nx_max_, ny_max_, nvx_max_, nvy_max_;

  pack(Config *conf, RealOffsetView4D halo_fn, Halos *send_halos)
    : conf_(conf), halo_fn_(halo_fn), send_halos_(send_halos) {
    buf_  = send_halos_->buf_;
    xmin_ = send_halos_->xmin_;
    xmax_ = send_halos_->xmax_;
    const Domain *dom = &(conf->dom_);
    nx_max_  = dom->nxmax_[0];
    ny_max_  = dom->nxmax_[1];
    nvx_max_ = dom->nxmax_[2];
    nvy_max_ = dom->nxmax_[3];
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
        const int jvy = ivy + ivy_min;
        const int ix_bc  = (nx_max_  + jx)  % nx_max_ ;
        const int iy_bc  = (ny_max_  + jy)  % ny_max_ ;
        const int ivx_bc = (nvx_max_ + jvx) % nvx_max_;
        const int ivy_bc = (nvy_max_ + jvy) % nvy_max_;
        int idx = Index::coord_4D2int(ix, iy, ivx, ivy, nx, ny, nvx, nvy);
        buf_(idx, ib) = halo_fn_(ix_bc, iy_bc, ivx_bc, ivy_bc);
      }
    }
  }
};

struct merged_pack {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  RealView1D buf_flatten_;
  RealView2D buf_;
  Halos      *send_halos_;
  IntView2D  flatten_map_;
   
  merged_pack(Halos *send_halos)
    : send_halos_(send_halos) {
    buf_flatten_ = send_halos_->buf_flatten_;
    buf_         = send_halos_->buf_;
    flatten_map_ = send_halos_->flatten_map_;
  }
   
  KOKKOS_INLINE_FUNCTION
  void operator()(const int idx) const {
    int idx_src = flatten_map_(idx, 0);
    int ib      = flatten_map_(idx, 1);
    buf_flatten_(idx) = buf_(idx_src, ib);
  }
};

struct merged_unpack {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  RealOffsetView4D halo_fn_;
  RealView1D       buf_flatten_;
  Halos            *recv_halos_;
  RangeView2D      map_;
   
  merged_unpack(RealOffsetView4D halo_fn, Halos *recv_halos)
    : halo_fn_(halo_fn), recv_halos_(recv_halos) {
    buf_flatten_ = recv_halos_->buf_flatten_;
    map_         = recv_halos_->map_;
  }
   
  KOKKOS_INLINE_FUNCTION
  void operator()(const int idx) const {
    const int ix  = map_(idx, 0), iy  = map_(idx, 1);
    const int ivx = map_(idx, 2), ivy = map_(idx, 3);
    halo_fn_(ix, iy, ivx, ivy) = buf_flatten_(idx);
  }
};

struct local_copy {
  typedef Kokkos::View<int*[DIMENSION], execution_space> RangeView2D;
  RealView1D  send_buf_, recv_buf_;
  Halos       *send_halos_, *recv_halos_;
  int         send_offset_, recv_offset_;
   
  local_copy(Halos *send_halos, Halos *recv_halos)
    : send_halos_(send_halos), recv_halos_(recv_halos) {
    send_buf_ = send_halos_->buf_flatten_;
    recv_buf_ = recv_halos_->buf_flatten_;
    send_offset_ = send_halos_->offset_local_copy_;
    recv_offset_ = recv_halos_->offset_local_copy_;
  }
   
  KOKKOS_INLINE_FUNCTION
  void operator()(const int idx) const {
    recv_buf_(idx+recv_offset_) = send_buf_(idx+send_offset_);
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
  Config           *conf_;
  RealOffsetView4D halo_fn_;
  RealView2D       buf_;
  Halos            *send_halos_;
  RangeView2D      xmin_, xmax_;
  RangeView2D      bc_in_min_, bc_in_max_;

  float64 alpha_;
  // Global domain size
  int nx_, ny_, nvx_, nvy_;

  // Local domain min and max
  int local_start_[4];
  int local_xstart_, local_ystart_, local_vxstart_, local_vystart_;

  // Pseudo constants
  int bc_sign_[8];

  boundary_condition(Config *conf, RealOffsetView4D halo_fn, Halos *send_halos)
    : conf_(conf), halo_fn_(halo_fn), send_halos_(send_halos) {
    buf_  = send_halos_->buf_;
    xmin_ = send_halos_->xmin_;
    xmax_ = send_halos_->xmax_;
    bc_in_min_ = send_halos_->bc_in_min_;
    bc_in_max_ = send_halos_->bc_in_max_;
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
                  fsum += halo_fn_(rx  + sign1[0] * j1, 
                                   ry  + sign1[1] * j1,
                                   rvx + sign1[2] * j1,
                                   rvy + sign1[3] * j1) * alphap1;
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
                        fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2, 
                                         ry  + sign1[1] * j1 + sign2[1] * j2,
                                         rvx + sign1[2] * j1 + sign2[2] * j2,
                                         rvy + sign1[3] * j1 + sign2[3] * j2) * alphap1;
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
                            fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2 + sign3[0] * j3, 
                                             ry  + sign1[1] * j1 + sign2[1] * j2 + sign3[1] * j3,
                                             rvx + sign1[2] * j1 + sign2[2] * j2 + sign3[2] * j3,
                                             rvy + sign1[3] * j1 + sign2[3] * j2 + sign3[3] * j3) * alphap1;
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
                                fsum += halo_fn_(rx  + sign1[0] * j1, 
                                                 ry  + sign2[1] * j2,
                                                 rvx + sign3[2] * j3,
                                                 rvy + sign4[3] * j4) * alphap1;
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
              fsum += halo_fn_(rx  + sign1[0] * j1, 
                               ry  + sign1[1] * j1,
                               rvx + sign1[2] * j1,
                               rvy + sign1[3] * j1) * alphap1;
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
                    fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2, 
                                     ry  + sign1[1] * j1 + sign2[1] * j2,
                                     rvx + sign1[2] * j1 + sign2[2] * j2,
                                     rvy + sign1[3] * j1 + sign2[3] * j2) * alphap1;
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
                        fsum += halo_fn_(rx  + sign1[0] * j1 + sign2[0] * j2 + sign3[0] * j3, 
                                         ry  + sign1[1] * j1 + sign2[1] * j2 + sign3[1] * j3,
                                         rvx + sign1[2] * j1 + sign2[2] * j2 + sign3[2] * j3,
                                         rvy + sign1[3] * j1 + sign2[3] * j2 + sign3[3] * j3) * alphap1;
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
                            fsum += halo_fn_(rx  + sign1[0] * j1, 
                                             ry  + sign2[1] * j2,
                                             rvx + sign3[2] * j3,
                                             rvy + sign4[3] * j4) * alphap1;
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

#endif
