#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include "../check.h"

#include "cutlass/gemm/device/gemm.h"

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
__global__ void matmul_share1(float* A, float* B, float* C, int M, int N, int K) {
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

void init_host_matrix(float* a, float* b, int M, int N, int K);
void compute_host_matrix(float* a, float* b, float* c, int M, int N, int K);
void check_matrix(float* c, float* host_c, int M, int N);

// ./test M N K kernel
int main(int argc, char** argv) {
    assert(argc == 5 && "./test M N K kernel");
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    int kernel = atoi(argv[4]);

    float *A, *B, *C;
    float* a = (float*)malloc(M * K * sizeof(float));
    float* b = (float*)malloc(K * N * sizeof(float));
    float* c = (float*)malloc(M * N * sizeof(float));
    float* host_c = (float*)malloc(M * N * sizeof(float));

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaMalloc((void**)&A, M * K * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&B, K * N * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&C, M * N * sizeof(float)));

    init_host_matrix(a, b, M, N, K);

    CUDACHECK(cudaMemcpy(A, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(B, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    CUDACHECK(cudaEventRecord(start, 0));
    switch (kernel) {
        case 0: {
            printf("M: %d, N: %d, K: %d, kernel: matmul\n", M, N, K);
            int width = 16;
            dim3 gridSize((M + width - 1) / width, (N + width - 1) / width);
            dim3 blockSize(width, width);
            matmul<<<gridSize, blockSize>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 1: {
            printf("M: %d, N: %d, K: %d, kernel: matmul1\n", M, N, K);
            int width = 16;
            dim3 gridSize((N + width - 1) / width, (M + width - 1) / width);
            dim3 blockSize(width, width);
            matmul1<<<gridSize, blockSize>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 2: {
            printf("M: %d, N: %d, K: %d, kernel: matmul_share\n", M, N, K);
            dim3 gridSize((M + TILE_WIDTH - 1) / TILE_WIDTH,
                          (N + TILE_WIDTH - 1) / TILE_WIDTH);
            dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
            matmul_share<<<gridSize, blockSize>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 3: {
            printf("M: %d, N: %d, K: %d, kernel: matmul_share1\n", M, N, K);
            dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH,
                          (M + TILE_WIDTH - 1) / TILE_WIDTH);
            dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
            matmul_share1<<<gridSize, blockSize>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 4: {
            printf("M: %d, N: %d, K: %d, kernel: cutlass_default\n", M, N, K);
            using RowMajor = cutlass::layout::RowMajor;
            using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;
            Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1.f, 0.f});
            Gemm gemm;
            CUTLASS_CHECK(gemm(args));
            after_kernel_launch();
            break;
        }
        default:
            printf("no such kernel.\n");
    }
    CUDACHECK(cudaEventRecord(stop, 0));
    CUDACHECK(cudaEventSynchronize(stop));
    // CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    CUDACHECK(cudaMemcpy(c, C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("time: %fms\n", elapsedTime);

    compute_host_matrix(a, b, host_c, M, N, K);
    check_matrix(c, host_c, M, N);

    free(a);
    free(b);
    free(c);
    free(host_c);
    CUDACHECK(cudaFree(A));
    CUDACHECK(cudaFree(B));
    CUDACHECK(cudaFree(C));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return 0;
}

void init_host_matrix(float* a, float* b, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = 1.f;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            b[i * N + j] = 2.f;
        }
    }
}

void compute_host_matrix(float* a, float* b, float* c, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float value = 0;
            for (int t = 0; t < K; t++) {
                value += a[i * K + t] * b[t * N + j];
            }
            c[i * N + j] = value;
        }
    }
}

void check_matrix(float* c, float* host_c, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (c[i * N + j] != host_c[i * N + j]) {
                fprintf(stderr,
                        "check failed: c[%d][%d], host: %f, device: %f\n", i, j,
                        host_c[i * N + j], c[i * N + j]);
                return;
            }
        }
    }
}
