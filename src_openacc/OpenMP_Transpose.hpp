#ifndef __OPENMP_TRANSPOSE_HPP__
#define __OPENMP_TRANSPOSE_HPP__

#include <omp.h>
#include <complex>
#include "Index.hpp"

template <typename RealType> using Complex = std::complex<RealType>;

namespace Impl {
  template <typename ScalarType, Layout LayoutType = Layout::LayoutRight, int blocksize = 16,
            typename std::enable_if<std::is_same<ScalarType, int             >::value ||
                                    std::is_same<ScalarType, float           >::value ||
                                    std::is_same<ScalarType, double          >::value ||
                                    std::is_same<ScalarType, Complex<float>  >::value ||
                                    std::is_same<ScalarType, Complex<double> >::value
                                   >::type * = nullptr>

  struct Transpose {
    private:
      int col_;
      int row_;
      const int blocksize_ = blocksize;
    public:
      typedef std::integral_constant<Layout, LayoutType> layout_;

    public:
      Transpose(int row, int col) {
        typedef std::integral_constant<Layout, Layout::LayoutLeft> layout_left;
        if(std::is_same<layout_, layout_left>::value) {
          row_ = row;
          col_ = col;
        } else {
          row_ = col;
          col_ = row;
        }
      }
      ~Transpose(){}

    public:
      // Interfaces
      void forward(ScalarType *dptr_in, ScalarType *dptr_out) {
        exec(dptr_in ,dptr_out, row_, col_);
      }

      void backward(ScalarType *dptr_in, ScalarType *dptr_out) {
        exec(dptr_in ,dptr_out, col_, row_);
      }

    private:
      void exec(ScalarType *dptr_in, ScalarType *dptr_out, int row, int col) {
        #pragma omp parallel for collapse(2)
        for(int j = 0; j < col; j += blocksize_) {
          for(int i = 0; i < row; i += blocksize_) {
            for(int c = j; c < j + blocksize_ && c < col; c++) {
              for(int r = i; r < i + blocksize_ && r < row; r++) {
                int idx_src = Index::coord_2D2int(r, c, row, col);
                int idx_dst = Index::coord_2D2int(c, r, col, row);
                dptr_out[idx_dst] = dptr_in[idx_src];
              }
            }
          }
        }
      }
  };
};

#endif
