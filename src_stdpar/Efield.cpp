#include "Efield.hpp"
#include <numeric>
#include <execution>
#include <algorithm>

Efield::Efield(Config *conf, shape_nd<2> dim)
  : fft_(nullptr) {
  rho_     = RealView2D("rho",     dim[0], dim[1]);
  rho_loc_ = RealView2D("rho_loc", dim[0], dim[1]);
  ex_      = RealView2D("ex",      dim[0], dim[1]);
  ey_      = RealView2D("ey",      dim[0], dim[1]);
  phi_     = RealView2D("phi",     dim[0], dim[1]);

  rho_.fill(0.);
  rho_loc_.fill(0.);
  ex_.fill(0.);
  ey_.fill(0.);
  phi_.fill(0.);

  // Initialize fft helper
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0];
  int ny = dom->nxmax_[1];
  float64 xmax = dom->maxPhy_[0];
  float64 kx0 = 2 * M_PI / xmax;

  fft_ = new Impl::FFT<float64, array_layout::value>(nx, ny, 1);
  int nx1h = fft_->nx1h();
  rho_hat_ = ComplexView2D("rho_hat", nx1h, ny);
  ex_hat_  = ComplexView2D("ex_hat",  nx1h, ny);
  ey_hat_  = ComplexView2D("ey_hat",  nx1h, ny);

  rho_hat_.fill(0.);
  ex_hat_.fill(0.);
  ey_hat_.fill(0.);

  filter_  = RealView1D("filter", nx1h);
  
  // Initialize filter (0,1/k2,1/k2,1/k2,...)
  // In order to avoid zero division in vectorized way
  // filter[0] == 0, and filter[0:] == 1./(ix1*kx0)**2
  filter_(0) = 0.;
  for(int ix = 1; ix < nx1h; ix++) {
    float64 kx = ix * kx0;
    float64 k2 = kx * kx;
    filter_(ix) = 1./k2;
  }
  filter_.updateDevice();
}

Efield::~Efield() {
  if(fft_ != nullptr) delete fft_;
}

void Efield::solve_poisson_fftw(float64 xmax, float64 ymax) {
  float64 kx0 = 2 * M_PI / xmax;
  float64 ky0 = 2 * M_PI / ymax;
  int nx1  = fft_->nx1();
  int nx1h = fft_->nx1h();
  int nx2  = fft_->nx2();
  int nx2h = fft_->nx2h();
  float64 normcoeff = fft_->normcoeff();
  const complex128 I = complex128(0., 1.);

  // Cast to raw pointers which are avilable in std::for_each_n
  auto idx_2d = rho_hat_.index();
  float64 *ptr_rho = rho_.data(), *ptr_filter = filter_.data();
  complex128 *ptr_ex_hat = ex_hat_.data(), *ptr_ey_hat = ey_hat_.data();
  complex128 *ptr_rho_hat = rho_hat_.data();

  // Define accessors by macro
  #define _filter(j0)      ptr_filter[j0]
  #define _ex_hat(j0, j1)  ptr_ex_hat[idx_2d(j0, j1)]
  #define _ey_hat(j0, j1)  ptr_ey_hat[idx_2d(j0, j1)]
  #define _rho_hat(j0, j1) ptr_rho_hat[idx_2d(j0, j1)]

  // Forward 2D FFT (Real to complex)
  fft_->rfft2(rho_.data(), rho_hat_.data());

  // Solve Poisson equation in Fourier space
  auto solve_poisson = [=](const int ix1) {
    float64 kx = ix1 * kx0;
    {
      int ix2 = 0;
      float64 kx = ix1 * kx0;
      _ex_hat(ix1, ix2) = -kx * I * _rho_hat(ix1, ix2) * _filter(ix1) * normcoeff;
      _ey_hat(ix1, ix2) = 0.;
      _rho_hat(ix1, ix2) = _rho_hat(ix1, ix2) * _filter(ix1) * normcoeff;
    }

    for(int ix2=1; ix2<nx2h; ix2++) {
      float64 ky = ix2 * ky0;
      float64 k2 = kx * kx + ky * ky;

      _ex_hat(ix1, ix2) = -(kx/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _ey_hat(ix1, ix2) = -(ky/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _rho_hat(ix1, ix2) = _rho_hat(ix1, ix2) / k2 * normcoeff;
    }

    for(int ix2=nx2h; ix2<nx2; ix2++) {
      float64 ky = (ix2-nx2) * ky0;
      float64 k2 = kx*kx + ky*ky;

      _ex_hat(ix1, ix2) = -(kx/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _ey_hat(ix1, ix2) = -(ky/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _rho_hat(ix1, ix2) = _rho_hat(ix1, ix2) / k2 * normcoeff;
    }
  };

  std::for_each_n(std::execution::par_unseq,
                  counting_iterator(0), nx1h,
                  solve_poisson);

  // Backward 2D FFT (Complex to Real)
  fft_->irfft2(rho_hat_.data(), rho_.data());
  fft_->irfft2(ex_hat_.data(),  ex_.data());
  fft_->irfft2(ey_hat_.data(),  ey_.data());
}
