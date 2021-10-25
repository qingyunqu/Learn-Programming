#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../check.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cublas_v2.h"

#include "mma.h"
using namespace nvcuda;

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define WARPSIZE 32

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

/*template <typename Ti, typename To>
__global__ void matmul(Ti* A, Ti* B, To* C, int M, int N, int K) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        To value = static_cast<To>(0);
        for (int k = 0; k < K; k++) {
            value += static_cast<To>(A[row * K + k] * B[k * N + col]);
        }
        C[row * N + col] = value;
    }
}
dim3 block(16, 16);
dim3 grid(CEIL_DIV(M, block.x), CEIL_DIV(N, block.y)); // row major: MxN
matmul<Ti, To><<<grid, block>>>(A, B, C, M, N, K);
*/

// M=1024, N=1024, K=1024  : 1.000160ms
template <typename Ti, typename To>
__global__ void matmul(Ti* A, Ti* B, To* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        To value = static_cast<To>(0);
        for (int k = 0; k < K; k++) {
            value += static_cast<To>(A[row * K + k] * B[k * N + col]);
        }
        C[row * N + col] = value;
    }
}

/*#define TILE_WIDTH 16
template <typename Ti, typename To>
__global__ void matmul_share(Ti* A, Ti* B, To* C, int M, int N, int K) {
    __shared__ Ti ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ Ti ds_B[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = by * TILE_WIDTH + ty;
    int row = bx * TILE_WIDTH + tx;

    To value = static_cast<To>(0);
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + ty < K) {
            ds_A[tx][ty] = A[row * K + t * TILE_WIDTH + ty];
        } else {
            ds_A[tx][ty] = static_cast<Ti>(0);
        }
        if (col < N && t * TILE_WIDTH + tx < K) {
            ds_B[tx][ty] = B[(t * TILE_WIDTH + tx) * N + col];
        } else {
            ds_B[tx][ty] = static_cast<Ti>(0);
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += static_cast<To>(ds_A[tx][k] * ds_B[k][ty]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}
dim3 block(TILE_WIDTH, TILE_WIDTH);
dim3 grid(CEIL_DIV(M, block.x), CEIL_DIV(N, block.y)); // row major: MxN
matmul_share<Ti, To><<<grid, block>>>(A, B, C, M, N, K);
*/

// M=1024, N=1024, K=1024  : 1.926304ms
#define TILE_M_1 16
#define TILE_N_1 16
#define TILE_K_1 16
template <typename Ti, typename To>
__global__ void matmul_tile(Ti* A, Ti* B, To* C, int M, int N, int K) {
    __shared__ Ti ds_A[TILE_M_1][TILE_K_1];
    __shared__ Ti ds_B[TILE_K_1][TILE_N_1];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_M_1 + ty;
    int col = bx * TILE_N_1 + tx;

    To value = static_cast<To>(0);
    for (int t = 0; t < CEIL_DIV(K, TILE_K_1); t++) {
        if (row < M && t * TILE_K_1 + tx < K) {
            ds_A[ty][tx] = A[row * K + (t * TILE_K_1 + tx)];
        } else {
            ds_A[ty][tx] = static_cast<Ti>(0);
        }
        if (col < N && t * TILE_K_1 + ty < K) {
            ds_B[ty][tx] = B[(t * TILE_K_1 + ty) * N + col];
        } else {
            ds_B[ty][tx] = static_cast<Ti>(0);
        }
        __syncthreads();
        for (int k = 0; k < TILE_K_1; k++) {
            value += static_cast<To>(ds_A[ty][k] * ds_B[k][tx]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// #define TILE_M_2 128
// #define TILE_N_2 128
// #define TILE_K_2 16
// template <typename Ti, typename To>
// __global__ void matmul_tile_megengine(Ti* A, Ti* B, To* C, int M, int N,
//                                       int K) {
//     __shared__ Ti ds_A[TILE_M_2][TILE_K_2];
//     __shared__ Ti ds_B[TILE_K_2][TILE_N_2];
//     int warp_id = threadIdx.x >> 5;
//     for(int k = 0; k < CEIL_DIV(K, TILE_K_2); k++) {

//     }
// }

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
__global__ void wmma_fp16(half* A, half* B, float* C, int M, int N, int K, float alpha, float beta) {
    int lda = K;
    int ldb = N;
    int ldc = N;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
  
    for (int k = 0; k < K; k += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = k;

      int bRow = k;
      int bCol = warpN * WMMA_N;

      if (aRow < M && aCol < K && bRow < K && bCol < N) {
        wmma::load_matrix_sync(a_frag, A + aRow * lda + aCol, lda);
        wmma::load_matrix_sync(b_frag, B + bRow * ldb + bCol, ldb);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, C + cRow * ldc + cCol, ldc, wmma::mem_row_major);
      // printf("c_frag.num_elements: %d\n", c_frag.num_elements);  8
      for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }
      wmma::store_matrix_sync(C + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}


template <typename T>
void init_matrix(T* a, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = static_cast<T>(rand() % 100);
        }
    }
}

template <typename T>
void check_matrix(T* c, T* ref_c, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (c[i * N + j] != ref_c[i * N + j]) {
                fprintf(stderr,
                        "check failed: c[%d][%d], ref: %f, kernel: %f\n", i, j,
                        static_cast<float>(ref_c[i * N + j]),
                        static_cast<float>(c[i * N + j]));
                return;
            }
        }
    }
}

// ./test M N K kernel
int main(int argc, char** argv) {
    assert(argc == 5 && "./test M N K kernel");
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    int kernel = atoi(argv[4]);

    CUDACHECK(cudaSetDevice(0));
    cudaEvent_t start, stop;
    float elapsedTime;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));

#define FP162FP32

#ifdef FP162FP16
    using Ti = half;
    using To = half;
#endif
#ifdef FP162FP32
    using Ti = half;
    using To = float;
#endif
#ifdef FP322FP32
    using Ti = float;
    using To = float;
#endif

    Ti *A, *B;
    To *C, *ref_C;
    CUDACHECK(cudaMalloc((void**)&A, M * K * sizeof(Ti)));
    CUDACHECK(cudaMalloc((void**)&B, K * N * sizeof(Ti)));
    CUDACHECK(cudaMalloc((void**)&C, M * N * sizeof(To)));
    CUDACHECK(cudaMalloc((void**)&ref_C, M * N * sizeof(To)));

    {
        Ti* a = (Ti*)malloc(M * K * sizeof(Ti));
        init_matrix<Ti>(a, M, K);
        CUDACHECK(cudaMemcpy(A, a, M * K * sizeof(Ti), cudaMemcpyHostToDevice));
        free(a);
    }
    {
        Ti* b = (Ti*)malloc(K * N * sizeof(Ti));
        init_matrix<Ti>(b, K, N);
        CUDACHECK(cudaMemcpy(B, b, K * N * sizeof(Ti), cudaMemcpyHostToDevice));
        free(b);
    }
    {
        using RowMajor = cutlass::layout::RowMajor;
        using ColMajor = cutlass::layout::ColumnMajor;
        using Gemm = cutlass::gemm::device::Gemm<Ti, RowMajor, Ti, RowMajor, To,
                                                 RowMajor>;
        Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {ref_C, N}, {ref_C, N},
                             {1.f, 0.f});
        Gemm gemm;
        CUTLASS_CHECK(gemm(args));
        after_kernel_launch();
    }

    CUDACHECK(cudaEventRecord(start, 0));
    switch (kernel) {
        case 0: {
            printf("M: %d, N: %d, K: %d, kernel: matmul           ", M, N, K);
            dim3 block(16, 16);
            dim3 grid(CEIL_DIV(N, block.x),
                      CEIL_DIV(M, block.y));  // row major: MxN
            matmul<Ti, To><<<grid, block>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 1: {
            printf("M: %d, N: %d, K: %d, kernel: matmul_tile(TILE_M: %d, "
                   "TILE_N: %d, TILE_K: %d) ",
                   M, N, K, TILE_M_1, TILE_N_1, TILE_K_1);
            dim3 block(TILE_N_1, TILE_M_1);
            dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
            matmul_tile<Ti, To><<<grid, block>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        // case 2: {
        //     printf("M: %d, N: %d, K: %d, kernel: matmul_tile_megengine(TILE_M: "
        //            "%d, "
        //            "TILE_N: %d, TILE_K: %d) ",
        //            M, N, K, TILE_M_2, TILE_N_2, TILE_K_2);
        //     dim3 block(128);
        //     dim3 grid(CEIL_DIV(N, TILE_N_2), CEIL_DIV(M, TILE_M_2));
        //     matmul_tile_megengine<Ti, To><<<grid, block>>>(A, B, C, M, N, K);
        //     after_kernel_launch();
        //     break;
        // }
        case 3: {
#ifdef FP162FP32
            printf("M: %d, N: %d, K: %d, kernel: wmma             ", M, N, K);
            dim3 block(128, 4);
            dim3 grid(CEIL_DIV(M, WMMA_M * block.x / WARPSIZE), CEIL_DIV(N, WMMA_N * block.y));
            wmma_fp16<<<grid, block>>>(A, B, C, M, N, K, 1.f, 0.f);
            after_kernel_launch();
#endif
            break;
        }
        case 4: {
            printf("M: %d, N: %d, K: %d, kernel: cutlass_default  ", M, N, K);
            using RowMajor = cutlass::layout::RowMajor;
            using ColMajor = cutlass::layout::ColumnMajor;
            using ThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 8>;  // <- threadblock tile M = 128, N = 128, K = 32
            using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;  // <- warp tile M = 32, N = 32, K = 32
            using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;  // <- MMA Op tile M = 8, N = 8, K = 4
            using Gemm = cutlass::gemm::device::Gemm<Ti, RowMajor,
                                                     Ti, RowMajor,
                                                     To, RowMajor,
                                                     To,
                                                     cutlass::arch::OpClassSimt,
                                                     cutlass::arch::Sm70,
                                                     ThreadBlockShape,
                                                     WarpShape,
                                                     InstructionShape>;
            Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N},
                                 {1.f, 0.f});
            Gemm gemm;
            CUTLASS_CHECK(gemm(args, nullptr, 0));
            break;
        }
        case 5: {
            // cublas normal is Column Major
            // CT = (AB)T = BT @ AT
            printf("M: %d, N: %d, K: %d, kernel: cublas           ", M, N, K);
#ifdef FP162FP16
            //__half alpha = static_cast<__half>(1.f), beta =
            // static_cast<__half>(0.f);
            __half alpha = __float2half(1.f), beta = __float2half(0.f);
            CUBLASCHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                    &alpha, B, N, A, K, &beta, C, N));
#endif
#ifdef FP162FP32
            float alpha = 1.f, beta = 0.f;
            /* IO in FP16/FP32, computation in float */
            CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                      &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F,
                                      K, &beta, C, CUDA_R_32F, N));
#endif
#ifdef FP322FP32
            float alpha = 1.f, beta = 0.f;
            CUBLASCHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                    &alpha, B, N, A, K, &beta, C, N));
#endif
            break;
        }
        default:
            printf("no such kernel.\n");
    }
    CUDACHECK(cudaEventRecord(stop, 0));
    CUDACHECK(cudaEventSynchronize(stop));
    // CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("time: %fms\n", elapsedTime);

    To* c = (To*)malloc(M * N * sizeof(To));
    To* ref_c = (To*)malloc(M * N * sizeof(To));
    CUDACHECK(cudaMemcpy(c, C, M * N * sizeof(To), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(ref_c, ref_C, M * N * sizeof(To),
                         cudaMemcpyDeviceToHost));

    check_matrix<To>(c, ref_c, M, N);

    free(c);
    free(ref_c);
    CUDACHECK(cudaFree(A));
    CUDACHECK(cudaFree(B));
    CUDACHECK(cudaFree(C));
    CUDACHECK(cudaFree(ref_C));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return 0;
}

// void check_matrix_col_row(float* c, float* ref_c, int M, int N) {
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             if (ref_c[i * N + j] != c[i + j * M]) {
//                 fprintf(stderr,
//                         "check failed: c[%d][%d], ref: %f, kernel: %f\n", i,
//                         j, ref_c[i * N + j], c[i + j * M]);
//                 return;
//             }
//         }
//     }
// }

// void check_matrix_col_col(float* c, float* ref_c, int M, int N) {
//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             if (ref_c[i + j * M] != c[i + j * M]) {
//                 fprintf(stderr,
//                         "check failed: c[%d][%d], ref: %f, kernel: %f\n", i,
//                         j, ref_c[i + j * M], c[i + j * M]);
//                 return;
//             }
//         }
//     }
// }
