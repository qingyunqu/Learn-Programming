#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include "../check.h"

#include "cutlass/gemm/device/gemm.h"

#include "cublas_v2.h"

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))

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

// M=1024, N=1024, K=1024  : 4.938048ms
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float value = 0.f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// M=1024, N=1024, K=1024  : 1.000160ms
__global__ void matmul1(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float value = 0.f;
        for (int k = 0; k < K; k++) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// M=1024, N=1024, K=1024  : 2.476992ms
#define TILE_WIDTH 16
__global__ void matmul_share(float* A, float* B, float* C, int M, int N,
                             int K) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = by * TILE_WIDTH + ty;
    int row = bx * TILE_WIDTH + tx;

    float value = 0.f;
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + ty < K) {
            ds_A[tx][ty] = A[row * K + t * TILE_WIDTH + ty];
        } else {
            ds_A[tx][ty] = 0.f;
        }
        if (col < N && t * TILE_WIDTH + tx < K) {
            ds_B[tx][ty] = B[(t * TILE_WIDTH + tx) * N + col];
        } else {
            ds_B[tx][ty] = 0.f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += ds_A[tx][k] * ds_B[k][ty];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// M=1024, N=1024, K=1024  : 1.926304ms
__global__ void matmul_share1(float* A, float* B, float* C, int M, int N,
                              int K) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float value = 0.f;
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + tx < K) {
            ds_A[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0.f;
        }
        if (col < N && t * TILE_WIDTH + ty < K) {
            ds_B[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            ds_B[ty][tx] = 0.f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

template <typename T>
__global__ void init_matrix(T* a, int row, int column, T value) {
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (r < row && c < column) {
        a[r * column + c] = value;
    }
}

void check_matrix(float* c, float* ref_c, int M, int N);
void check_matrix_col_row(float* c, float* ref_c, int M, int N);

// ./test M N K kernel
int main(int argc, char** argv) {
    assert(argc == 5 && "./test M N K kernel");
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    int kernel = atoi(argv[4]);

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    cublasHandle_t handle;
    CUBLASCHECK(cublasCreate(&handle));

    float *A, *B, *C, *ref_C;
    CUDACHECK(cudaMalloc((void**)&A, M * K * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&B, K * N * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&C, M * N * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&ref_C, M * N * sizeof(float)));

    {
        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(K, block.x), CEIL_DIV(M, block.y));
        init_matrix<float><<<grid, block>>>(A, M, K, 1.f);
    }
    {
        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(K, block.y));
        init_matrix<float><<<grid, block>>>(B, K, N, 2.f);
    }
    {
        using RowMajor = cutlass::layout::RowMajor;
        using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float,
                                                 RowMajor, float, RowMajor>;
        Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {ref_C, N}, {ref_C, N},
                             {1.f, 0.f});
        Gemm gemm;
        CUTLASS_CHECK(gemm(args));
        after_kernel_launch();
    }

    CUDACHECK(cudaEventRecord(start, 0));
    switch (kernel) {
        case 0: {
            printf("M: %d, N: %d, K: %d, kernel: matmul          ", M, N, K);
            dim3 block(16, 16);
            dim3 grid(CEIL_DIV(M, block.x), CEIL_DIV(N, block.y));
            matmul<<<grid, block>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 1: {
            printf("M: %d, N: %d, K: %d, kernel: matmul1         ", M, N, K);
            dim3 block(16, 16);
            dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
            matmul1<<<grid, block>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 2: {
            printf("M: %d, N: %d, K: %d, kernel: matmul_share    ", M, N, K);
            dim3 block(TILE_WIDTH, TILE_WIDTH);
            dim3 grid(CEIL_DIV(M, block.x), CEIL_DIV(N, block.y));
            matmul_share<<<grid, block>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 3: {
            printf("M: %d, N: %d, K: %d, kernel: matmul_share1   ", M, N, K);
            dim3 block(TILE_WIDTH, TILE_WIDTH);
            dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
            matmul_share1<<<grid, block>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 4: {
            printf("M: %d, N: %d, K: %d, kernel: cutlass_default ", M, N, K);
            using RowMajor = cutlass::layout::RowMajor;
            using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float,
                                                     RowMajor, float, RowMajor>;
            Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N},
                                 {1.f, 0.f});
            Gemm gemm;
            CUTLASS_CHECK(gemm(args));
            after_kernel_launch();
            break;
        }
        case 5: {
            printf("M: %d, N: %d, K: %d, kernel: cublas ", M, N, K);
            float alpha = 1.f, beta = 0.f;
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, A, K,
                        B, N, &beta, C, M);
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

    float* c = (float*)malloc(M * N * sizeof(float));
    float* ref_c = (float*)malloc(M * N * sizeof(float));
    CUDACHECK(cudaMemcpy(c, C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(ref_c, ref_C, M * N * sizeof(float),
                         cudaMemcpyDeviceToHost));

    if (kernel != 5)
        check_matrix(c, ref_c, M, N);
    else
        check_matrix_col_row(c, ref_c, M, N);

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

void check_matrix(float* c, float* ref_c, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (c[i * N + j] != ref_c[i * N + j]) {
                fprintf(stderr,
                        "check failed: c[%d][%d], ref: %f, kernel: %f\n", i, j,
                        ref_c[i * N + j], c[i * N + j]);
                return;
            }
        }
    }
}

void check_matrix_col_row(float* c, float* ref_c, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (ref_c[i * N + j] != c[i + j * M]) {
                fprintf(stderr,
                        "check failed: c[%d][%d], ref: %f, kernel: %f\n", i, j,
                        ref_c[i * N + j], c[i + j * M]);
                return;
            }
        }
    }
}
