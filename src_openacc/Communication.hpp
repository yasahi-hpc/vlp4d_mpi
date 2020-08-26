#ifndef __COMM_HPP__
#define __COMM_HPP__

#include <vector>
#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <string>
#include "Types.hpp"
#include "Config.hpp"
#include "Index.hpp"
#include "Timer.hpp"
#include "Helper.hpp"

static constexpr int VUNDEF = -100000;

// Element considered within the algorithm of the Unbalanced Recursive Bisection (URB)
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
  typedef View<int, 2, array_layout::value> RangeView2D;
  RealView2D buf_;
  RealView1D buf_flatten_;
  RangeView2D xmin_;
  RangeView2D xmax_;
  RangeView2D bc_in_min_;
  RangeView2D bc_in_max_;
  RangeView2D lxmin_;
  RangeView2D lxmax_;

  shape_nd<DIMENSION> nhalo_max_;
  int size_;     // buffer size of each halo
  int nb_halos_; // the number of halos
  int pid_;      // MPI rank
  int nbp_;      // the number of MPI processes

  std::vector<int> sizes_;
  std::vector<int> pids_;
  std::vector<int> tags_;

  /* Used for merge */
  RangeView2D map_;         // f -> flatten_buf
  IntView2D   flatten_map_; // buf -> flatten_buf
  int offset_local_copy_;   // The head address for the local copy
  std::vector<int> merged_sizes_; // buffer size of each halo
  std::vector<int> merged_pids_; // merged process id to interact with
  std::vector<int> merged_tags_; //
  std::vector<int> id_map_; // mapping the original id to the merged id
  std::vector<int> merged_heads_;
  int total_size_; // total size of all the buffers
  int nb_merged_halos_; // number of halos after merge

public:
  // Constructor and destructor
  Halos() {
    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this)
    #endif
  }

  ~Halos() {
    #if defined( ENABLE_OPENACC )
      #pragma acc exit data delete(this)
    #endif
  }

public:
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
     
  int nb_reqs(){ return (nbp_ - 1); }
  int nb_halos(){ return nb_halos_; } 

  void mergeLists(Config *conf, const std::string name) {
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
      int tag = tags_[same_dst[0]]; // Use the tag of first element
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
    map_         = RangeView2D("map", total_size, DIMENSION); // This is used for receive buffer
    buf_flatten_ = RealView1D(name + "_buf_flat", total_size);
    flatten_map_ = IntView2D("flatten_map", total_size, 2); // storing (idx_in_buf, buf_id)

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
    for(auto same_dst: group_same_dst) {
      // Keeping the head index of each halo sets for MPI communication
      merged_heads_.push_back(idx_flatten);
      for(auto it: same_dst) {
        const int ix_min  = xmin_(it, 0), ix_max  = xmax_(it, 0);
        const int iy_min  = xmin_(it, 1), iy_max  = xmax_(it, 1);
        const int ivx_min = xmin_(it, 2), ivx_max = xmax_(it, 2);
        const int ivy_min = xmin_(it, 3), ivy_max = xmax_(it, 3);
                                                                   
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
                map_(idx_flatten, 0) = ix;
                map_(idx_flatten, 1) = iy;
                map_(idx_flatten, 2) = ivx;
                map_(idx_flatten, 3) = ivy;
                                                                                                 
                // h_flatten_map is used for send buffer
                flatten_map_(idx_flatten, 0) = idx;
                flatten_map_(idx_flatten, 1) = it;
                idx_flatten++;
              }
            }
          }
        }
      }
    }

    // Deep copy
    map_.updateDevice();
    flatten_map_.updateDevice();
  }

  void set(Config *conf, std::vector<Halo> &list, const std::string name, const int nb_process, const int pid) {
    nbp_ = nb_process;
    pid_ = pid;
    nb_merged_halos_ = nbp_;

    nb_halos_ = list.size();
    pids_.resize(nb_halos_);
    tags_.resize(nb_halos_);

    xmin_      = RangeView2D("halo_xmin",  nb_halos_, DIMENSION);
    xmax_      = RangeView2D("halo_xmax",  nb_halos_, DIMENSION);
    lxmin_     = RangeView2D("halo_lxmin", nb_halos_, DIMENSION);
    lxmax_     = RangeView2D("halo_lxmax", nb_halos_, DIMENSION);
    bc_in_min_ = RangeView2D("bc_in_min",  nb_halos_, DIMENSION);
    bc_in_max_ = RangeView2D("bc_in_max",  nb_halos_, DIMENSION);

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
        xmin_(i, j)  = halo->xmin_[j]; 
        xmax_(i, j)  = halo->xmax_[j];
        lxmin_(i, j) = halo->lxmin_[j]; 
        lxmax_(i, j) = halo->lxmax_[j]; 
        int lxmin = lxmin_(i, j) - HALO_PTS, lxmax = lxmax_(i, j) + HALO_PTS;
        bc_in_min_(i, j) = (xmin_(i, j) <= lxmin && lxmin <= xmax_(i, j)) ? lxmin : VUNDEF;
        bc_in_max_(i, j) = (xmin_(i, j) <= lxmax && lxmax <= xmax_(i, j)) ? lxmax : VUNDEF;
      }
    }

    xmin_.updateDevice();
    xmax_.updateDevice();
    bc_in_min_.updateDevice();
    bc_in_max_.updateDevice();

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

struct Distrib {
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
  Halos *send_buffers_; // May be better to use pointer for explicit deallocation

  // List of halo buffers (sending side)
  Urbnode *node_;

  // Use spline or not
  bool spline_;

  // Global domain size
  int nxmax_[DIMENSION];

public:
  Distrib() = delete;
  Distrib(int &argc, char **argv) : spline_(true), recv_buffers_(nullptr), send_buffers_(nullptr) {
    int required = MPI_THREAD_SERIALIZED;
    int provided;

    // Initialize MPI
    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp_);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid_);

    // set GPUs
    #if defined( ENABLE_OPENACC )
      const int ngpus = acc_get_num_devices(acc_device_nvidia);
      acc_set_device_num(pid_ % ngpus, acc_device_nvidia);
      #pragma acc enter data copyin(this)
    #endif
  };

  ~Distrib(){
    #if defined( ENABLE_OPENACC )
      #pragma acc exit data delete(this)
    #endif
  }

  // Deallocate pointer inside the Kokkos parallel region
  void cleanup() {
    #if defined( ENABLE_OPENACC )
      #pragma acc exit data delete(send_buffers_[0:1], recv_buffers_[0:1]) 
    #endif
    delete recv_buffers_;
    delete send_buffers_;
  }

  void finalize() {
    MPI_Finalize();
  }

  // Getters
  bool master() {return pid_ == 0;}
  int pid(){return pid_;}
  int rank(){return pid_;}
  int nbp(){return nbp_;}
  Urbnode *node(){return node_;}
  std::vector<Urbnode> &nodes(){return ulist_;}

  // Initializers
  void createDecomposition(Config *conf);
  void neighboursList(Config *conf, RealView4D &halo_fn);
  void bookHalo(Config *conf);

  // Communication
  void exchangeHalo(Config *conf, RealView4D &halo_fn, std::vector<Timer*> &timers);

private:
  void getNeighbours(const Config *conf, const RealView4D &halo_fn, int xrange[8],
                     std::vector<Halo> &hlist, int lxmin[4], int lxmax[4], int count);

  void fillHalo(const Config *conf, RealView4D halo_fn);
  void fillHaloBoundaryCond(const Config *conf, RealView4D halo_fn);
  void fillHaloBoundaryCondOrig(const Config *conf, RealView4D halo_fn);

  void packAndBoundary(Config *conf, RealView4D &halo_fn, Halos *send_buffers);

  void pack(Config *conf, RealView4D &halo_fn, Halos *send_buffers) {
    const Domain *dom = &(conf->dom_);
    const int nx_max  = dom->nxmax_[0];
    const int ny_max  = dom->nxmax_[1];
    const int nvx_max = dom->nxmax_[2];
    const int nvy_max = dom->nxmax_[3];
    const int nb_halos = send_buffers->nb_halos();
    const int nx  = halo_fn.strides(0);
    const int ny  = halo_fn.strides(1);
    const int nvx = halo_fn.strides(2);
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop collapse(4) present(this,halo_fn,send_buffers->xmin_,send_buffers->xmax_,send_buffers->buf_)
    #else
      #pragma omp parallel for collapse(3)
    #endif
    for(int ib=0; ib<nb_halos; ib++) {
      for(int ivx=0; ivx<nvx; ivx++) {
        for(int iy=0; iy<ny; iy++) {
          for(int ix=0; ix<nx; ix++) {
            const int ix_min  = send_buffers->xmin_(ib, 0), ix_max  = send_buffers->xmax_(ib, 0);
            const int iy_min  = send_buffers->xmin_(ib, 1), iy_max  = send_buffers->xmax_(ib, 1);
            const int ivx_min = send_buffers->xmin_(ib, 2), ivx_max = send_buffers->xmax_(ib, 2);
            const int ivy_min = send_buffers->xmin_(ib, 3), ivy_max = send_buffers->xmax_(ib, 3);

            const int nx_tmp  = ix_max  - ix_min  + 1;
            const int ny_tmp  = iy_max  - iy_min  + 1;
            const int nvx_tmp = ivx_max - ivx_min + 1;
            const int nvy_tmp = ivy_max - ivy_min + 1;

            const int jx  = ix  + ix_min;
            const int jy  = iy  + iy_min;
            const int jvx = ivx + ivx_min;

            if( (jx <= ix_max) && (jy <= iy_max) && (jvx <= ivx_max) ) {
              for(int ivy = 0; ivy < nvy_tmp; ivy++) {
                // Pack into halo->buf as a 1D flatten array
                // Periodice boundary condition in each direction
                const int jvy = ivy + ivy_min;
                const int ix_bc = (nx_max + jx) % nx_max;
                const int iy_bc = (ny_max + jy) % ny_max;
                const int ivx_bc = (nvx_max + jvx) % nvx_max;
                const int ivy_bc = (nvy_max + jvy) % nvy_max;
                int idx = Index::coord_4D2int(ix, iy, ivx, ivy,
                                              nx_tmp, ny_tmp, nvx_tmp, nvy_tmp);
                send_buffers->buf_(idx, ib) = halo_fn(ix_bc, iy_bc, ivx_bc, ivy_bc);
              }
            }
          }
        }
      }
    }
  }

  // Called after applying boundary condition
  void merged_pack(Halos *send_buffers) {
    const int total_size = send_buffers->total_size_;
    //std::cout << "total_size" << total_size << std::endl;
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop present(this,send_buffers->flatten_map_, send_buffers->buf_, send_buffers->buf_flatten_)
    #else
      #pragma omp parallel for
    #endif
    for(int idx=0; idx<total_size; idx++) {
      int idx_src = send_buffers->flatten_map_(idx, 0);
      int ib      = send_buffers->flatten_map_(idx, 1);
      send_buffers->buf_flatten_(idx) = send_buffers->buf_(idx_src, ib);
    }
  }

  void unpack(RealView4D &halo_fn, Halos *recv_buffers) {
    const int total_size = recv_buffers->total_size_;
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop present(this,halo_fn, recv_buffers->map_, recv_buffers->buf_flatten_)
    #else
      #pragma omp parallel for
    #endif
    for(int idx=0; idx<total_size; idx++) {
      const int ix  = recv_buffers->map_(idx, 0), iy  = recv_buffers->map_(idx, 1);
      const int ivx = recv_buffers->map_(idx, 2), ivy = recv_buffers->map_(idx, 3);
      halo_fn(ix, iy, ivx, ivy) = recv_buffers->buf_flatten_(idx);
    }
  };

  void local_copy(Halos *send_buffers, Halos *recv_buffers) {
    const int send_offset = send_buffers->offset_local_copy_;
    const int recv_offset = recv_buffers->offset_local_copy_;
    const int total_size  = send_buffers->merged_size(pid_);
    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop present(this,send_buffers->buf_flatten_, recv_buffers->buf_flatten_)
    #else
      #pragma omp parallel for
    #endif
    for(int idx=0; idx<total_size; idx++) {
      recv_buffers->buf_flatten_(idx+recv_offset) = send_buffers->buf_flatten_(idx + send_offset);
    }
  }

  // Boundary condition
  void boundary_condition(Config *conf, RealView4D &halo_fn, Halos *send_halos) {
    //#if defined( ENABLE_OPENACC )
    //  boundary_condition_acc_(conf, halo_fn, send_halos);
    //#else
    //  boundary_condition_omp_(conf, halo_fn, send_halos);
    //#endif
      boundary_condition_omp_(conf, halo_fn, send_halos);
  }

  // OpenMP specialization (loop over each halo)
  void boundary_condition_omp_(Config *conf, RealView4D &halo_fn, Halos *send_halos) {
    const Domain *dom = &(conf->dom_);
    int nx = dom->nxmax_[0], ny = dom->nxmax_[1], nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
    int bc_sign[8];
    for(int k = 0; k < DIMENSION; k++) {
      bc_sign[2 * k + 0] = -1;
      bc_sign[2 * k + 1] = 1;
    }
    float64 alpha = sqrt(3) - 2;
    const int nb_send_halos = send_halos->nb_halos();

    #if defined( ENABLE_OPENACC )
      #pragma acc parallel loop gang present(this,send_halos->xmin_, send_halos->xmax_, \
                                             send_halos->bc_in_min_, send_halos->bc_in_max_, \
                                             halo_fn, send_halos->buf_)
    #else
      #pragma omp parallel for
    #endif
    for(int ib = 0; ib < nb_send_halos; ib++) {
      int halo_min[4], halo_max[4];
      for(int k = 0; k < DIMENSION; k++) {
        halo_min[k] = send_halos->xmin_(ib, k);
        halo_max[k] = send_halos->xmax_(ib, k);
      }

      const int halo_nx  = halo_max[0] - halo_min[0] + 1;
      const int halo_ny  = halo_max[1] - halo_min[1] + 1;
      const int halo_nvx = halo_max[2] - halo_min[2] + 1;
      const int halo_nvy = halo_max[3] - halo_min[3] + 1;

      int bc_in[8], orcheck[4];
      int orcsum = 0;
      char bitconf = 0;

      for(int k = 0; k < 4; k++) {
        bc_in[2 * k + 0] = send_halos->bc_in_min_(ib, k);
        bc_in[2 * k + 1] = send_halos->bc_in_max_(ib, k);
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
              sign1[ii] = bc_sign[k1], vdx[ii] = bc_in[k1], vex[ii] = bc_in[k1];
          }

          for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
            for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
              for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                  int rx  = (nx  + jx  - sign1[0]) % nx;
                  int ry  = (ny  + jy  - sign1[1]) % ny;
                  int rvx = (nvx + jvx - sign1[2]) % nvx;
                  int rvy = (nvy + jvy - sign1[3]) % nvy;
                  float64 fsum = 0.;
                  float64 alphap1 = alpha;
                  for(int j1 = 1; j1 <= MMAX; j1++) {
                    fsum += halo_fn(rx  + sign1[0] * j1,
                                    ry  + sign1[1] * j1,
                                    rvx + sign1[2] * j1,
                                    rvy + sign1[3] * j1) * alphap1;
                                                                                                                                        
                    alphap1 *= alpha;
                  }
                  int idx = Index::coord_4D2int(jx - halo_min[0],
                                                jy  - halo_min[1],
                                                jvx - halo_min[2], 
                                                jvy - halo_min[3], 
                                                halo_nx, halo_ny, halo_nvx, halo_nvy);
                                  
                  send_halos->buf_(idx, ib) = fsum;
                } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
              } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
            } // for(int jvx = vdx[2]; jvx <= vex[2]; jvx++)
          } // for(int jvy = vdx[3]; jvy <= vex[3]; jvy++)
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
                  sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                if(ii == k2/2)
                  sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
              }

              for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                  for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                    for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                      int rx  = (nx  + jx  - sign1[0] - sign2[0]) % nx;
                      int ry  = (ny  + jy  - sign1[1] - sign2[1]) % ny;
                      int rvx = (nvx + jvx - sign1[2] - sign2[2]) % nvx;
                      int rvy = (nvy + jvy - sign1[3] - sign2[3]) % nvy;

                      float64 fsum = 0.;
                      float64 alphap2 = alpha;
                      for(int j2 = 1; j2 <= MMAX; j2++) {
                        float64 alphap1 = alpha * alphap2;
                        for(int j1 = 1; j1 <= MMAX; j1++) {
                          fsum += halo_fn(rx  + sign1[0] * j1 + sign2[0] * j2, 
                                          ry  + sign1[1] * j1 + sign2[1] * j2,
                                          rvx + sign1[2] * j1 + sign2[2] * j2,
                                          rvy + sign1[3] * j1 + sign2[3] * j2) * alphap1;
                                                  
                          alphap1 *= alpha;
                        }
                        alphap2 *= alpha;
                      }
                      int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                    jy  - halo_min[1],
                                                    jvx - halo_min[2],
                                                    jvy - halo_min[3],
                                                    halo_nx, halo_ny, halo_nvx, halo_nvy);
                                          
                      send_halos->buf_(idx, ib) = fsum;
                    } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
                  } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
                } // for(int jvx = vdx[2]; jvx <= vex[2]; jvx++)
              } // for(int jvy = vdx[3]; jvy <= vex[3]; jvy++)
            } // for(int k1 = 0; k1 < 8; k1++)
          } // for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++)
        } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF) 
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
                    sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                  if(ii == k2/2)
                    sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
                  if(ii == k3/2)
                    sign3[ii] = bc_sign[k3], vex[ii] = vdx[ii] = bc_in[k3];
                }

                for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                  for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                    for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                      for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                        int rx  = (nx  + jx  - sign1[0] - sign2[0] - sign3[0]) % nx;
                        int ry  = (ny  + jy  - sign1[1] - sign2[1] - sign3[1]) % ny;
                        int rvx = (nvx + jvx - sign1[2] - sign2[2] - sign3[2]) % nvx;
                        int rvy = (nvy + jvy - sign1[3] - sign2[3] - sign3[3]) % nvy;
                        float64 fsum = 0.;
                        float64 alphap3 = alpha;
                        for(int j3 = 1; j3 <= MMAX; j3++) {
                          float64 alphap2 = alpha * alphap3;
                          for(int j2 = 1; j2 <= MMAX; j2++) {
                            float64 alphap1 = alpha * alphap2;
                            for(int j1 = 1; j1 <= MMAX; j1++) {
                              fsum += halo_fn(rx  + sign1[0] * j1 + sign2[0] * j2 + sign3[0] * j3,
                                              ry  + sign1[1] * j1 + sign2[1] * j2 + sign3[1] * j3,
                                              rvx + sign1[2] * j1 + sign2[2] * j2 + sign3[2] * j3,
                                              rvy + sign1[3] * j1 + sign2[3] * j2 + sign3[3] * j3) * alphap1;
                              alphap1 *= alpha;
                            }
                            alphap2 *= alpha;
                          }
                          alphap3 *= alpha;
                        } // for(int j3 = 1; j3 <= MMAX; j3++)
                        int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                      jy  - halo_min[1],
                                                      jvx - halo_min[2],
                                                      jvy - halo_min[3],
                                                      halo_nx, halo_ny, halo_nvx, halo_nvy);
                        send_halos->buf_(idx, ib) = fsum;
                      } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
                    } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
                  } // for(int jvx = vdx[2]; jvx <= vex[2]; jvx++)
                } // for(int jvy = vdx[3]; jvy <= vex[3]; jvy++)
              } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF)
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
                      sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                    if(ii == k2/2)
                      sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
                    if(ii == k3/2)
                      sign3[ii] = bc_sign[k3], vex[ii] = vdx[ii] = bc_in[k3];
                    if(ii == k4/2)
                      sign4[ii] = bc_sign[k4], vex[ii] = vdx[ii] = bc_in[k4];
                  }

                  for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                    for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                      for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                        for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                          int rx  = (nx  + jx  - sign1[0]) % nx;
                          int ry  = (ny  + jy  - sign2[1]) % ny;
                          int rvx = (nvx + jvx - sign3[2]) % nvx;
                          int rvy = (nvy + jvy - sign4[3]) % nvy;
                          float64 fsum = 0.;
                          float64 alphap4 = alpha;
                          for(int j4 = 1; j4 <= MMAX; j4++) {
                            float64 alphap3 = alpha * alphap4;
                            for(int j3 = 1; j3 <= MMAX; j3++) {
                              float64 alphap2 = alpha * alphap3;
                              for(int j2 = 1; j2 <= MMAX; j2++) {
                                float64 alphap1 = alpha * alphap2;
                                for(int j1 = 1; j1 <= MMAX; j1++) {
                                  fsum += halo_fn(rx  + sign1[0] * j1, 
                                                  ry  + sign2[1] * j2,
                                                  rvx + sign3[2] * j3,
                                                  rvy + sign4[3] * j4) * alphap1;
                                                                
                                  alphap1 *= alpha;
                                }
                                alphap2 *= alpha;
                              }
                              alphap3 *= alpha;
                            }
                            alphap4 *= alpha;
                          }
                          int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                        jy  - halo_min[1],
                                                        jvx - halo_min[2],
                                                        jvy - halo_min[3],
                                                        halo_nx, halo_ny, halo_nvx, halo_nvy);
                          send_halos->buf_(idx, ib) = fsum;
                        } // for(int jx = vdx[0]; jx <= vex[0]; jx++)
                      } // for(int jy = vdx[1]; jy <= vex[1]; jy++)
                    } // for(int jvx = vdx[2]; jvx <= vex[2]; jvx++)
                  } // for(int jvy = vdx[3]; jvy <= vex[3]; jvy++)
                } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF && bc_in[k4] != VUNDEF)
              } // for(int k4 = 6; k4 < 8; k4++)
            } // for(int k3 = 4; k3 < 6; k3++)
          } // for(int k2 = 2; k2 < 4; k2++)
        } // for(int k1 = 0; k1 < 2; k1++)
      } // if(orcsum > 3)
    } // for(int ib = 0; ib < nb; ib++)
  }

  void boundary_condition_acc_(Config *conf, RealView4D &halo_fn, Halos *send_halos) {
    const Domain *dom = &(conf->dom_);
    int nx = dom->nxmax_[0], ny = dom->nxmax_[1], nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
    int bc_sign[8];
    for(int k = 0; k < DIMENSION; k++) {
      bc_sign[2 * k + 0] = -1;
      bc_sign[2 * k + 1] = 1;
    }
    float64 alpha = sqrt(3) - 2;
    const int nb_send_halos = send_halos->nb_halos();
    const int nx_send  = send_halos->nhalo_max_[0];
    const int ny_send  = send_halos->nhalo_max_[1];
    const int nvx_send = send_halos->nhalo_max_[2];

    std::cout << "nx_send, ny_send, nvx_send, nb_send_halos = "
              << nx_send << ", " << ny_send << ", " << nvx_send << ", " << nb_send_halos << std::endl;

    #pragma acc parallel loop collapse(4) present(this,send_halos->xmin_, send_halos->xmax_, \
                                                  send_halos->bc_in_min_, send_halos->bc_in_max_, \
                                                  halo_fn, send_halos->buf_)
    for(int ib = 0; ib < nb_send_halos; ib++) {
      for(int ivx = 0; ivx < nvx_send; ivx++) {
        for(int iy = 0; iy < ny_send; iy++) {
          for(int ix = 0; ix < nx_send; ix++) {
            int halo_min[4], halo_max[4];
            for(int k = 0; k < DIMENSION; k++) {
              halo_min[k] = send_halos->xmin_(ib, k);
              halo_max[k] = send_halos->xmax_(ib, k);
            }
            const int halo_nx  = halo_max[0] - halo_min[0] + 1;
            const int halo_ny  = halo_max[1] - halo_min[1] + 1;
            const int halo_nvx = halo_max[2] - halo_min[2] + 1;
            const int halo_nvy = halo_max[3] - halo_min[3] + 1;

            int bc_in[8], orcheck[4];
            int orcsum = 0;
            char bitconf = 0;
            for(int k = 0; k < 4; k++) {
              bc_in[2 * k + 0] = send_halos->bc_in_min_(ib, k);
              bc_in[2 * k + 1] = send_halos->bc_in_max_(ib, k);
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
                    sign1[ii] = bc_sign[k1], vdx[ii] = bc_in[k1], vex[ii] = bc_in[k1];
                  }
                }

                const int jx  = ix  + vdx[0];
                const int jy  = iy  + vdx[1];
                const int jvx = ivx + vdx[2];
                if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
                  for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                    int rx  = (nx  + jx  - sign1[0]) % nx;
                    int ry  = (ny  + jy  - sign1[1]) % ny;
                    int rvx = (nvx + jvx - sign1[2]) % nvx;
                    int rvy = (nvy + jvy - sign1[3]) % nvy;
                    float64 fsum = 0.;
                    float64 alphap1 = alpha;
                    for(int j1 = 1; j1 <= MMAX; j1++) {
                      fsum += halo_fn(rx  + sign1[0] * j1,
                                      ry  + sign1[1] * j1,
                                      rvx + sign1[2] * j1,
                                      rvy + sign1[3] * j1) * alphap1;
                      alphap1 *= alpha;
                    }
                    int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                  jy  - halo_min[1],
                                                  jvx - halo_min[2], 
                                                  jvy - halo_min[3], 
                                                  halo_nx, halo_ny, halo_nvx, halo_nvy);

                    send_halos->buf_(idx, ib) = fsum;
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
                        sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                      if(ii == k2/2)
                        sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
                    }

                    const int jx  = ix  + vdx[0];
                    const int jy  = iy  + vdx[1];
                    const int jvx = ivx + vdx[2];
                    if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
                      int rx  = (nx  + jx  - sign1[0] - sign2[0]) % nx;
                      int ry  = (ny  + jy  - sign1[1] - sign2[1]) % ny;
                      int rvx = (nvx + jvx - sign1[2] - sign2[2]) % nvx;
                      for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                        int rvy = (nvy + jvy - sign1[3] - sign2[3]) % nvy;
                        float64 fsum = 0.;
                        float64 alphap2 = alpha;
                        for(int j2 = 1; j2 <= MMAX; j2++) {
                          float64 alphap1 = alpha * alphap2;  
                          for(int j1 = 1; j1 <= MMAX; j1++) {
                            fsum += halo_fn(rx  + sign1[0] * j1 + sign2[0] * j2,
                                            ry  + sign1[1] * j1 + sign2[1] * j2,
                                            rvx + sign1[2] * j1 + sign2[2] * j2,
                                            rvy + sign1[3] * j1 + sign2[3] * j2) * alphap1;
                            alphap1 *= alpha;
                          }
                          alphap2 *= alpha;
                        }
                        int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                      jy  - halo_min[1],
                                                      jvx - halo_min[2],
                                                      jvy - halo_min[3],
                                                      halo_nx, halo_ny, halo_nvx, halo_nvy);
                        send_halos->buf_(idx, ib) = fsum;
                      } // for(int jvy = vdx[3]; jvy <= vex[3]; jvy++)
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
                          sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                                                                                                                                  
                        if(ii == k2/2)
                          sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
                                                                                                                                                  
                        if(ii == k3/2)
                          sign3[ii] = bc_sign[k3], vex[ii] = vdx[ii] = bc_in[k3];
                      }

                      const int jx  = ix  + vdx[0];
                      const int jy  = iy  + vdx[1];
                      const int jvx = ivx + vdx[2];
                      if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
                        int rx  = (nx  + jx  - sign1[0] - sign2[0] - sign3[0]) % nx;
                        int ry  = (ny  + jy  - sign1[1] - sign2[1] - sign3[1]) % ny;
                        int rvx = (nvx + jvx - sign1[2] - sign2[2] - sign3[2]) % nvx;
                        for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                          int rvy = (nvy + jvy - sign1[3] - sign2[3] - sign3[3]) % nvy;
                          float64 fsum = 0.;
                          float64 alphap3 = alpha;
                          for(int j3 = 1; j3 <= MMAX; j3++) {
                            float64 alphap2 = alpha * alphap3;
                            for(int j2 = 1; j2 <= MMAX; j2++) {
                              float64 alphap1 = alpha * alphap2;
                              for(int j1 = 1; j1 <= MMAX; j1++) {
                                fsum += halo_fn(rx  + sign1[0] * j1 + sign2[0] * j2 + sign3[0] * j3,
                                                ry  + sign1[1] * j1 + sign2[1] * j2 + sign3[1] * j3,
                                                rvx + sign1[2] * j1 + sign2[2] * j2 + sign3[2] * j3,
                                                rvy + sign1[3] * j1 + sign2[3] * j2 + sign3[3] * j3) * alphap1;
                                alphap1 *= alpha;
                              }
                              alphap2 *= alpha;
                            }
                            alphap3 *= alpha;
                          }
                          int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                        jy  - halo_min[1],
                                                        jvx - halo_min[2],
                                                        jvy - halo_min[3],
                                                        halo_nx, halo_ny, halo_nvx, halo_nvy);
                          send_halos->buf_(idx, ib) = fsum;
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
                            sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                          if(ii == k2/2)
                            sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
                          if(ii == k3/2)
                            sign3[ii] = bc_sign[k3], vex[ii] = vdx[ii] = bc_in[k3];
                          if(ii == k4/2)
                            sign4[ii] = bc_sign[k4], vex[ii] = vdx[ii] = bc_in[k4];
                        }

                        const int jx  = ix  + vdx[0];
                        const int jy  = iy  + vdx[1];
                        const int jvx = ivx + vdx[2];
                        if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] ) {
                          int rx  = (nx  + jx  - sign1[0]) % nx;
                          int ry  = (ny  + jy  - sign2[1]) % ny;
                          int rvx = (nvx + jvx - sign3[2]) % nvx;
                          for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                            int rvy = (nvy + jvy - sign4[3]) % nvy;
                            float64 fsum = 0.;
                            float64 alphap4 = alpha;
                            for(int j4 = 1; j4 <= MMAX; j4++) {
                              float64 alphap3 = alpha * alphap4;
                              for(int j3 = 1; j3 <= MMAX; j3++) {
                                float64 alphap2 = alpha * alphap3;
                                for(int j2 = 1; j2 <= MMAX; j2++) {
                                  float64 alphap1 = alpha * alphap2;
                                  for(int j1 = 1; j1 <= MMAX; j1++) {
                                    fsum += halo_fn(rx  + sign1[0] * j1,
                                                    ry  + sign2[1] * j2,
                                                    rvx + sign3[2] * j3,
                                                    rvy + sign4[3] * j4) * alphap1;
                                    alphap1 *= alpha;
                                  }
                                  alphap2 *= alpha;
                                }
                                alphap3 *= alpha;
                              }
                              alphap4 *= alpha;
                            }
                            int idx = Index::coord_4D2int(jx  - halo_min[0], 
                                                          jy  - halo_min[1],
                                                          jvx - halo_min[2],
                                                          jvy - halo_min[3],
                                                          halo_nx, halo_ny, halo_nvx, halo_nvy);
                                                
                            send_halos->buf_(idx, ib) = fsum;
                          } // for(int ivy = 0; ivy < tmp_nvy; ivy++)
                        } // if( jx <= vex[0] && jy <= vex[1] && jvx <= vex[2] )
                      } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
                    } // for(int k4 = 6; k4 < 8; k4++)
                  } // for(int k3 = 4; k3 < 6; k3++)
                } // for(int k2 = 2; k2 < 4; k2++)
              } // for(int k1 = 0; k1 < 8; k1++)
            } // if(orcsum > 3)
          } // for(int ix = 0; ix < nx_send; ix++)
        } // for(int iy = 0; iy < ny_send; iy++)
      } // for(int ivx = 0; ivx < nvx_send; ivx++)
    } // for(int ib = 0; ib < nb_send_halos; ib++)
  }
            
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
  }
};

#endif
