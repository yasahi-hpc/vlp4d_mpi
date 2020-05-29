#ifndef __CUDA_TRANSPOSE_HPP__
#define __CUDA_TRANSPOSE_HPP__

#include <cublas_v2.h>
#include "types.h"

namespace Impl {
  template < typename ScalarType >
  struct Transpose {
    private:
      int col_;
      int row_;
      cublasHandle_t handle_;

    public:

      Transpose(int row, int col) : row_(row), col_(col) {
        cublasCreate(&handle_);
      }

      virtual ~Transpose() {
        cublasDestroy(handle_);
      }

      /*
      // In-place transpose
      void forward(ScalarType *dptr_inout,
                   typename std::enable_if<std::is_same<ScalarType, float32>::value   ||
                                           std::is_same<ScalarType, float64>::value   ||
                                           std::is_same<ScalarType, complex32>::value ||
                                           std::is_same<ScalarType, complex64>::value >::type * = nullptr) {
        cublasTranspose_(dptr_inout, dptr_inout, row_, col_);
      }

      void backward(ScalarType *dptr_inout,
                   typename std::enable_if<std::is_same<ScalarType, float32>::value   ||
                                           std::is_same<ScalarType, float64>::value   ||
                                           std::is_same<ScalarType, complex32>::value ||
                                           std::is_same<ScalarType, complex64>::value >::type * = nullptr) {
        cublasTranspose_(dptr_inout, dptr_inout, col_, row_);
      }
      */

      // Out-place transpose
      void forward(ScalarType *dptr_in, ScalarType *dptr_out,
                   typename std::enable_if<std::is_same<ScalarType, float32>::value   ||
                                           std::is_same<ScalarType, float64>::value   ||
                                           std::is_same<ScalarType, complex32>::value ||
                                           std::is_same<ScalarType, complex64>::value >::type * = nullptr) {
        cublasTranspose_(dptr_in, dptr_out, row_, col_);
      }

      void backward(ScalarType *dptr_in, ScalarType *dptr_out,
                    typename std::enable_if<std::is_same<ScalarType, float32>::value   ||
                                            std::is_same<ScalarType, float64>::value   ||
                                            std::is_same<ScalarType, complex32>::value ||
                                            std::is_same<ScalarType, complex64>::value >::type * = nullptr) {
        cublasTranspose_(dptr_in, dptr_out, col_, row_);
      }

    private:

      // Float32 specialization
      void cublasTranspose_(float32 *dptr_in, float32 *dptr_out, int row, int col) {
        const float32 alpha = 1.0;
        const float32 beta  = 0.0;
        cublasSgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_T, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    dptr_in,     // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    dptr_in,     // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    dptr_out,    // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // Float64 specialization
      void cublasTranspose_(float64 *dptr_in, float64 *dptr_out, int row, int col) {
        const float64 alpha = 1.;
        const float64 beta  = 0.;
        cublasDgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_T, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    dptr_in,     // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    dptr_in,     // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    dptr_out,    // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // complex32 specialization
      void cublasTranspose_(complex32 *dptr_in, complex32 *dptr_out, int row, int col) {
        const cuComplex alpha = make_cuComplex(1.0, 0.0);
        const cuComplex beta  = make_cuComplex(0.0, 0.0);
        cublasCgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_N, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    reinterpret_cast<cuComplex*>(dptr_in), // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    reinterpret_cast<cuComplex*>(dptr_in), // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    reinterpret_cast<cuComplex*>(dptr_out), // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // complex64 specialization
      void cublasTranspose_(complex64 *dptr_in, complex64 *dptr_out, int row, int col) {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1., 0.);
        const cuDoubleComplex beta  = make_cuDoubleComplex(0., 0.);
        cublasZgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_N, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    reinterpret_cast<cuDoubleComplex*>(dptr_in),  // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    reinterpret_cast<cuDoubleComplex*>(dptr_in),  // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    reinterpret_cast<cuDoubleComplex*>(dptr_out), // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }
  };
}

#endif
