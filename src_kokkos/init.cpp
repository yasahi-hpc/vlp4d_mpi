#include "init.h"
#include "helper.hpp"

// Prototypes
void import(const char *f, Config *conf);
void print(Config *conf);
void initcase(Config *conf, RealView4D fn);
void testcaseCYL02(Config* conf, RealView4D fn);
void testcaseCYL05(Config* conf, RealView4D fn);
void testcaseSLD10(Config* conf, RealView4D fn);
void testcaseTSI20(Config* conf, RealView4D fn);

void import(const char *f, Config *conf) {
  char idcase[8], tmp;
  Physics* phys = &(conf->phys_);
  Domain* dom = &(conf->dom_);
  FILE* stream = fopen(f, "r");

  if(stream == (FILE*)NULL) {
    printf("import: error file not found\n");
    abort();
  }

  /*Physical parameters */
  phys->eps0_ = 1.; /* permittivity of free space */
  phys->echarge_ = 1.; /* charge of particle */
  phys->mass_ = 1; /* mass of particle */
  phys->omega02_ = 0; /* coefficient of applied field */
  phys->vbeam_ = 0.; /* beam velocity */
  phys->S_ = 1; /* lattice period */
  phys->psi_ = 0;

  for(int i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %d\n", &(dom->nxmax_[i]));
  }

  for(int i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom->minPhy_[i]));
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom->maxPhy_[i]));
  }

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fgets(idcase, 7, stream);

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %le\n", &(dom->dt_));

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->nbiter_));
  
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->ifreq_));
   
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->fxvx_));

  dom->idcase_ = atoi(idcase);
  
  for(int i = 0; i < DIMENSION; i++) {
    dom->dx_[i] = (dom->maxPhy_[i] - dom->minPhy_[i]) / dom->nxmax_[i];
  }
  
  fclose(stream);
};

void print(Config *conf) {
  Domain* dom = &(conf->dom_);

  printf("** Definition of mesh\n");
  printf("Number of points in  x with the coarse mesh : %d\n", dom->nxmax_[0]);
  printf("Number of points in  y with the coarse mesh : %d\n", dom->nxmax_[1]);
  printf("Number of points in Vx with the coarse mesh : %d\n", dom->nxmax_[2]);
  printf("Number of points in Vy with the coarse mesh : %d\n", dom->nxmax_[3]);
  
  printf("\n** Defintion of the geometry of the domain\n");
  printf("Minimal value of Ex : %lf\n", dom->minPhy_[0]);
  printf("Maximal value of Ex : %lf\n", dom->maxPhy_[0]);
  printf("Minimal value of Ey : %lf\n", dom->minPhy_[1]);
  printf("Maximal value of Ey : %lf\n", dom->maxPhy_[1]);
  
  printf("\nMinimal value of Vx : %lf\n", dom->minPhy_[2]);
  printf("Maximal value of Vx   : %lf\n", dom->maxPhy_[2]);
  printf("Minimal value of Vy   : %lf\n", dom->minPhy_[3]);
  printf("Maximal value of Vy   : %lf\n", dom->maxPhy_[3]);
  
  printf("\n** Considered test cases");
  printf("\n-10- Landau Damping");
  printf("\n-11- Landau Damping 2");
  printf("\n-20- Two beam instability");
  printf("\nNumber of the chosen test case : %d\n", dom->idcase_);
  
  printf("\n** Iterations in time and diagnostics\n");
  printf("Time step : %lf\n", dom->dt_);
  printf("Number of total iterations : %d\n", dom->nbiter_);
  printf("Frequency of diagnostics : %d\n", dom->ifreq_);
  
  printf("Diagnostics of fxvx : %d\n", dom->fxvx_);
}

void initcase(Config* conf, RealView4D fn) {
  Domain* dom = &(conf->dom_);
 
  switch(dom->idcase_) {
    case 2:
      testcaseCYL02(conf, fn);
      break; // CYL02 ;
    case 5:
      testcaseCYL05(conf, fn);
      break; // CYL05 ;
    case 10:
      testcaseSLD10(conf, fn);
      break; // SLD10 ;
    case 20:
      testcaseTSI20(conf, fn);
      break; // TSI20 ;
    default:
      printf("Unknown test case !\n");
      abort();
      break;
  }
}

void testcaseCYL02(Config* conf, RealView4D fn) {
  Domain* dom = &(conf->dom_);
  const float64 PI = M_PI;
  const float64 AMPLI = 4;
  const float64 PERIOD = 0.5 * PI;
  const float64 cc = 0.50 * (6. / 16.);
  const float64 rc = 0.50 * (4. / 16.);

  typename RealView4D::HostMirror h_fn = Kokkos::create_mirror_view(fn);

  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
      float64 vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          float64 x = dom->minPhy_[0] + ix * dom->dx_[0];
          float64 xx = x;
          float64 vv = vx;
          
          float64 hv = 0.0;
          float64 hx = 0.0;
          
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          } else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }
          
          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          } else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
          
          int jx  = ix  - dom->local_nxmin_[0] + HALO_PTS;
          int jy  = iy  - dom->local_nxmin_[1] + HALO_PTS;
          int jvx = ivx - dom->local_nxmin_[2] + HALO_PTS;
          int jvy = ivy - dom->local_nxmin_[3] + HALO_PTS;
          h_fn(jx, jy, jvx, jvy) = (AMPLI * hx * hv);
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

void testcaseCYL05(Config* conf, RealView4D fn) {
  Domain * dom = &(conf->dom_);
  const float64 PI = M_PI;
  const float64 AMPLI = 4;
  const float64 PERIOD = 0.5 * PI;
  const float64 cc = 0.50 * (6. / 16.);
  const float64 rc = 0.50 * (4. / 16.);

  typename RealView4D::HostMirror h_fn = Kokkos::create_mirror_view(fn);
  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx < dom->local_nxmax_[2]; ivx++) {
      float64 vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      for(int iy = dom->local_nxmin_[1]; iy < dom->local_nxmax_[1]; iy++) {
        float64 y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = dom->local_nxmin_[0]; ix < dom->local_nxmax_[0]; ix++) {
          float64 xx = y;
          float64 vv = vy;
          
          float64 hv = 0.0;
          float64 hx = 0.0;
          
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          } else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }
          
          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          } else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
          
          int jx  = ix  - dom->local_nxmin_[0] + HALO_PTS;
          int jy  = iy  - dom->local_nxmin_[1] + HALO_PTS;
          int jvx = ivx - dom->local_nxmin_[2] + HALO_PTS;
          int jvy = ivy - dom->local_nxmin_[3] + HALO_PTS;
          h_fn(jx, jy, jvx, jvy) = AMPLI * hx * hv;
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

void testcaseSLD10(Config* conf, RealView4D fn) {
  Domain * dom = &(conf->dom_);

  typename RealView4D::HostMirror h_fn = Kokkos::create_mirror_view(fn);
  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
      float64 vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      float64 vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
        float64 y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          float64 x = dom->minPhy_[0] + ix * dom->dx_[0];
            
          float64 sum = (vx * vx + vy * vy);
          int jx  = ix  - dom->local_nxmin_[0] + HALO_PTS;
          int jy  = iy  - dom->local_nxmin_[1] + HALO_PTS;
          int jvx = ivx - dom->local_nxmin_[2] + HALO_PTS;
          int jvy = ivy - dom->local_nxmin_[3] + HALO_PTS;
          h_fn(jx, jy, jvx, jvy) = (1. / (2 * M_PI)) * exp(-0.5 * (sum)) * (1 + 0.05 * (cos(0.5 * x) * cos(0.5 * y)));
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

void testcaseTSI20(Config* conf, RealView4D fn) {
  Domain * dom = &(conf->dom_);

  float64 vd  = 2.4;
  float64 vthx  = 1.0;
  float64 vthy  = 0.5;
  float64 alphax  = 0.05;
  float64 alphay  = 0.25;
  float64 kx  = 0.2;
  float64 ky  = 0.2;
    
  typename RealView4D::HostMirror h_fn = Kokkos::create_mirror_view(fn);
  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
      double vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      double vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          double yy = dom->minPhy_[1] + iy * dom->dx_[1];
          double xx = dom->minPhy_[0] + ix * dom->dx_[0];

          int jx  = ix  - dom->local_nxmin_[0] + HALO_PTS;
          int jy  = iy  - dom->local_nxmin_[1] + HALO_PTS;
          int jvx = ivx - dom->local_nxmin_[2] + HALO_PTS;
          int jvy = ivy - dom->local_nxmin_[3] + HALO_PTS;
          h_fn(jx, jy, jvx, jvy) = 
            1./(4*M_PI*vthx*vthy)*
            (1-alphax*sin(kx*xx)-alphay*sin(ky*yy))*
            (exp(-.5*(pow((vx-vd)/vthx,2)+pow(vy/vthy,2)))+
                exp(-.5*(pow((vx+vd)/vthx,2)+pow(vy/vthy,2))));
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

/* @brief pack some numbers into halo_fn for checking the communication routines
 *        node->xmin_[0], node->xmax_[0] = 16, 31
 *        node->xmin_[1], node->xmax_[1] = 0, 31
 *        node->xmin_[2], node->xmax_[2] = 0, 31
 *        node->xmin_[3], node->xmax_[3] = 0, 31
 *        node->xmin_[0], node->xmax_[0] = 0, 15
 *        node->xmin_[1], node->xmax_[1] = 0, 31
 *        node->xmin_[2], node->xmax_[2] = 0, 31
 *        node->xmin_[3], node->xmax_[3] = 0, 31
 */
void testcase_ptest_init(Config *conf, Distrib &comm, RealView4D halo_fn) {
  const Domain *dom = &(conf->dom_);
  Urbnode *node = comm.node();

  typename RealView4D::HostMirror h_halo_fn = Kokkos::create_mirror_view(halo_fn);

  for(int ivy = node->xmin_[3]; ivy <= node->xmax_[3]; ivy++) {
    for(int ivx = node->xmin_[2]; ivx <= node->xmax_[2]; ivx++) {
      for(int iy = node->xmin_[1]; iy <= node->xmax_[1]; iy++) {
        for(int ix = node->xmin_[0]; ix <= node->xmax_[0]; ix++) {
          h_halo_fn(ix  + HALO_PTS - dom->local_nxmin_[0],
                    iy  + HALO_PTS - dom->local_nxmin_[1], 
                    ivx + HALO_PTS - dom->local_nxmin_[2],
                    ivy + HALO_PTS - dom->local_nxmin_[3]) = (double)((ivy + 111 * ivx) * 111 + iy) * 111 + ix;
        }
      }
    }
  }

  Kokkos::deep_copy(halo_fn, h_halo_fn);
}

void testcase_ptest_check(Config* conf, Distrib &comm, RealView4D halo_fn) {
  Domain *dom = &(conf->dom_);
  Urbnode *node = comm.node();
  int offp = HALO_PTS - 1, offm = HALO_PTS - 1;

  typename RealView4D::HostMirror h_halo_fn = Kokkos::create_mirror_view(halo_fn);
  Kokkos::deep_copy(h_halo_fn, halo_fn);
  for(int ivy = node->xmin_[3] - offm; ivy <= node->xmax_[3] + offp; ivy++) {
    for(int ivx = node->xmin_[2] - offm; ivx <= node->xmax_[2] + offp; ivx++) {
      const int jvy = (dom->nxmax_[3] + ivy) % dom->nxmax_[3];
      const int jvx = (dom->nxmax_[2] + ivx) % dom->nxmax_[2];
   
      for(int iy = node->xmin_[1] - offm; iy <= node->xmax_[1] + offp; iy++) {
        const int jy = (dom->nxmax_[1] + iy) % dom->nxmax_[1];
                                                                                                                          
        for(int ix = node->xmin_[0] - offm; ix <= node->xmax_[0] + offp; ix++) {
          const int jx = (dom->nxmax_[0] + ix) % dom->nxmax_[0];
          double fval = h_halo_fn(ix  - node->xmin_[0] + HALO_PTS, 
                                  iy  - node->xmin_[1] + HALO_PTS, 
                                  ivx - node->xmin_[2] + HALO_PTS, 
                                  ivy - node->xmin_[3] + HALO_PTS);
          double ref = (((double)jvy + 111. * (double)jvx) * 111. + (double)jy) * 111. + (double)jx;
          double diff = fval - ref;
          if(fabs(diff) > .1) {
            printf("[%d] Pb %d %d %d %d: %lf %lf %lf\n", comm.pid(), ivy, ivx, iy, ix, fval, ref, diff);
          } else {
            //printf("[%d] OK %d %d %d %d: %lf %lf\n", comm.pid(),ivy,ivx,iy,ix, fval,ref);
          }
        }
      }
    }
  }
}

void init(const char *file, Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, Efield **ef, Diags **dg, std::vector<Timer*> &timers) {
  Domain* dom = &conf->dom_; 

  import(file, conf);
  if(comm.master()) print(conf);

  // Initialize communication manager
  comm.createDecomposition(conf);
  Urbnode *mynode = comm.node();

  shape_t<DIMENSION> shape_halo;
  for(int i=0; i<DIMENSION; i++)
    shape_halo[i] = mynode->xmax_[i] - mynode->xmin_[i] + 2 * HALO_PTS + 1;

  // Allocate 4D data structures
  int nx = shape_halo[0], ny = shape_halo[1], nvx = shape_halo[2], nvy = shape_halo[3];
  fn   = RealView4D("fn",   nx, ny, nvx, nvy);
  fnp1 = RealView4D("fnp1", nx, ny, nvx, nvy);

  comm.neighboursList(conf, fn);
  comm.bookHalo(conf);

  // The functions testcase_ptest_init and testcase_ptest_check are present
  // to check the good behavior of comm_exchange_halo. These calls are also
  // there to initiate a first round of communication as a startup phase.
  testcase_ptest_init(conf, comm, fn);
  comm.exchangeHalo(conf, fn, timers);
  testcase_ptest_check(conf, comm, fn);

  timers[Halo_fill_A]->reset();
  timers[Halo_comm]->reset();
  timers[Halo_fill_B]->reset();

  *ef = new Efield(conf, {dom->nxmax_[0], dom->nxmax_[1]});

  // allocate and initialize diagnostics data structures
  *dg = new Diags(conf);

  // Initialize distribution function
  fn = RealView4D("fn",   nx, ny, nvx, nvy); // Needed for CPU to zero init
  initcase(conf, fn);
}

void finalize(Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, Efield **ef, Diags **dg) {
  // Store diagnostics
  delete *ef;
  delete *dg;
}
