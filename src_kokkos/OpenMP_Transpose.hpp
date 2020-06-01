#ifndef __CPU_TRANSPOSE_H__
#define __CPU_TRANSPOSE_H__

#include "types.h"
#include <omp.h>

/*
 * Simple wrapper for transpoes
 * For CPU, LayoutRight (C style is assumed)
 */

namespace Impl {
  template < typename ScalarType >
  struct Transpose {
    private:
      int col_;
      int row_;
      const int blocksize_ = 16;

    public:

      Transpose(int col, int row) : row_(row), col_(col) {
      }

      virtual ~Transpose() {
      }

      /*
      // In-place transpose
      void exec(ScalarType *dptr_inout,
                typename std::enable_if<std::is_same<ScalarType, int>::value       ||
                                        std::is_same<ScalarType, float32>::value   ||
                                        std::is_same<ScalarType, float64>::value   ||
                                        std::is_same<ScalarType, complex32>::value ||
                                        std::is_same<ScalarType, complex64>::value >::type * = nullptr) {
          // Not implemented yet
      }
      */
    public:
      // Out-place transpose
      void forward(ScalarType *dptr_in, ScalarType *dptr_out,
                   typename std::enable_if<std::is_same<ScalarType, int>::value       ||
                                           std::is_same<ScalarType, float32>::value   ||
                                           std::is_same<ScalarType, float64>::value   ||
                                           std::is_same<ScalarType, complex64>::value ||
                                           std::is_same<ScalarType, complex128>::value >::type * = nullptr) {
          //exec_serial(dptr_in, dptr_out, row_, col_);
          exec(dptr_in, dptr_out, row_, col_);
      }

      void backward(ScalarType *dptr_in, ScalarType *dptr_out,
                    typename std::enable_if<std::is_same<ScalarType, int>::value       ||
                                            std::is_same<ScalarType, float32>::value   ||
                                            std::is_same<ScalarType, float64>::value   ||
                                            std::is_same<ScalarType, complex64>::value ||
                                            std::is_same<ScalarType, complex128>::value >::type * = nullptr) {
          //exec_serial(dptr_in, dptr_out, row_, col_);
          exec(dptr_in, dptr_out, col_, row_);
      }


    private:
      // Out-place transpose
      void exec(ScalarType *dptr_in, ScalarType *dptr_out, int row, int col,
                typename std::enable_if<std::is_same<ScalarType, int>::value       ||
                                        std::is_same<ScalarType, float32>::value   ||
                                        std::is_same<ScalarType, float64>::value   ||
                                        std::is_same<ScalarType, complex64>::value ||
                                        std::is_same<ScalarType, complex128>::value >::type * = nullptr) {
        #pragma omp parallel for schedule(static) collapse(2)
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

      void exec_serial(ScalarType *dptr_in, ScalarType *dptr_out, int row, int col,
                typename std::enable_if<std::is_same<ScalarType, int>::value       ||
                                        std::is_same<ScalarType, float32>::value   ||
                                        std::is_same<ScalarType, float64>::value   ||
                                        std::is_same<ScalarType, complex64>::value ||
                                        std::is_same<ScalarType, complex128>::value >::type * = nullptr) {
         for(int c = 0; c < col; c++) {
           for(int r = 0; r < row; r++) {
             int idx_src = Index::coord_2D2int(r, c, row, col);
             int idx_dst = Index::coord_2D2int(c, r, col, row);
             dptr_out[idx_dst] = dptr_in[idx_src];
           }
         }
      }
  };
}

#endif
