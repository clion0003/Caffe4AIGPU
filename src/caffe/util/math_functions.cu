#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  __global__ void mat_mul_N_N(int m, int k ,int n, float alpha, float beta, const float *a, const float *b, float *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    float result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[row*k + index] * b[index*n + col];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  __global__ void mat_mul_N_T(int m, int k ,int n, float alpha, float beta, const float *a, const float *b, float *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    float result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[row*k + index] * b[col*k + index];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  __global__ void mat_mul_T_N(int m, int k ,int n, float alpha, float beta, const float *a, const float *b, float *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    float result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[index*m + row] * b[index*n + col];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  __global__ void mat_mul_T_T(int m, int k ,int n, float alpha, float beta, const float *a, const float *b, float *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    float result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[index*m + row] * b[col*k + index];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  
  __global__ void mat_vec_N(int m, int n, float alpha, float beta, const float *a, const float *x ,float *y) {
    // calculate the row & col index of the element
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    float result = 0;
    if (row < m) {
      // do dot product between row of a and col of b
      for (int index = 0; index < n; index++)
        result += a[row*n + index] * x[index];
  
      y[row] = alpha * result + beta * y[row];
  
    }
  }
  
  __global__ void mat_vec_T(int m, int n, float alpha, float beta, const float *a, const float *x ,float *y) {
    // calculate the row & col index of the element
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    float result = 0;
    if (row < m) {
      // do dot product between row of a and col of b
      for (int index = 0; index < n; index++)
        result += a[index*m + row] * x[index];
  
      y[row] = alpha * result + beta * y[row];
  
    }
  }

  __global__ void double_mat_mul_N_N(int m, int k ,int n, double alpha, double beta, const double *a, const double *b, double *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    double result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[row*k + index] * b[index*n + col];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  __global__ void double_mat_mul_N_T(int m, int k ,int n, double alpha, double beta, const double *a, const double *b, double *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    double result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[row*k + index] * b[col*k + index];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  __global__ void double_mat_mul_T_N(int m, int k ,int n, double alpha, double beta, const double *a, const double *b, double *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    double result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[index*m + row] * b[index*n + col];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  __global__ void double_mat_mul_T_T(int m, int k ,int n, double alpha, double beta, const double *a, const double *b, double *ab) {
    // calculate the row & col index of the element
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    double result = 0;
    if (row < m && col < n) {
      // do dot product between row of a and col of b
      for (int index = 0; index < k; index++)
        result += a[index*m + row] * b[col*k + index];
  
      ab[row*n + col] = alpha * result + beta * ab[row*n + col];
  
    }
  }
  
  
  __global__ void double_mat_vec_N(int m, int n, double alpha, double beta, const double *a, const double *x ,double *y) {
    // calculate the row & col index of the element
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    double result = 0;
    if (row < m) {
      // do dot product between row of a and col of b
      for (int index = 0; index < n; index++)
        result += a[row*n + index] * x[index];
  
      y[row] = alpha * result + beta * y[row];
  
    }
  }
  
  __global__ void double_mat_vec_T(int m, int n, double alpha, double beta, const double *a, const double *x ,double *y) {
    // calculate the row & col index of the element
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    double result = 0;
    if (row < m) {
      // do dot product between row of a and col of b
      for (int index = 0; index < n; index++)
        result += a[index*m + row] * x[index];
  
      y[row] = alpha * result + beta * y[row];
  
    }
  }

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  // int lda = (TransA == CblasNoTrans) ? K : M;
  // int ldb = (TransB == CblasNoTrans) ? N : K;
  //cublasOperation_t cuTransA =
  //    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  //cublasOperation_t cuTransB =
  //    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
      
	dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  if(TransA == CblasNoTrans){
    if(TransB == CblasNoTrans) mat_mul_N_N << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
    else mat_mul_N_T << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
  }
  else{
    if(TransB == CblasNoTrans) mat_mul_T_N << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
    else mat_mul_T_T << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
  }
  //CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
  //    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
	dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  if(TransA == CblasNoTrans){
    if(TransB == CblasNoTrans) double_mat_mul_N_N << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
    else double_mat_mul_N_T << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
  }
  else{
    if(TransB == CblasNoTrans) double_mat_mul_T_N << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
    else double_mat_mul_T_T << <grid, block >> > (M, K, N, alpha, beta, A, B, C);
  }
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  //cublasOperation_t cuTransA =
  //    (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  //CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
  //    A, N, x, 1, &beta, y, 1));
	dim3 block(256);
  dim3 grid((N + 255) / 256);
  if(TransA == CblasNoTrans) mat_vec_N <<< grid, block >>> (M, N, alpha, beta, A, x, y);
  else mat_vec_T <<< grid, block >>> (M, N, alpha, beta, A, x, y);
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
//  cublasOperation_t cuTransA =
//      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
//      A, N, x, 1, &beta, y, 1));
dim3 block(256);
dim3 grid((N + 255) / 256);
if(TransA == CblasNoTrans) double_mat_vec_N <<< grid, block >>> (M, N, alpha, beta, A, x, y);
else double_mat_vec_T <<< grid, block >>> (M, N, alpha, beta, A, x, y);
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  //CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  //CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  //CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  //CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  //cudaStream_t initial_stream;
  //CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  //CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  //CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  //CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  //cudaStream_t initial_stream;
  //CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  //CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  //CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  //CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  //CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  //CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  //CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  //CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  //CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  //CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  //CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  //CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  //CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  //CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  //const float range = b - a;
  //if (range != static_cast<float>(1)) {
  //  caffe_gpu_scal(n, range, r);
  //}
  //if (a != static_cast<float>(0)) {
  //  caffe_gpu_add_scalar(n, a, r);
  //}
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  //CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  //const double range = b - a;
  //if (range != static_cast<double>(1)) {
  //  caffe_gpu_scal(n, range, r);
  //}
  //if (a != static_cast<double>(0)) {
  //  caffe_gpu_add_scalar(n, a, r);
  //}
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  //CURAND_CHECK(
  //    curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  //CURAND_CHECK(
  //    curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
