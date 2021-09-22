#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include "../check.h"

// M=1024, N=1024, K=1024  : 1.010144ms
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float value = 0.f;
        for (int i = 0; i < N; i++) {
            value += A[row * N + i] * B[col + i * K];
        }
        C[row * K + col] = value;
        // printf("row: %d, col: %d, value: %f\n", row, col, value);
    }
}

// M=1024, N=1024, K=1024  : 1.907552ms
#define TILE_WIDTH 16
__global__ void matmul_share(float* A, float* B, float* C, int M, int N,
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
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + tx < N) {
            ds_A[tx][ty] = A[row * N + t * TILE_WIDTH + tx];
        } else {
            ds_A[tx][ty] = 0.f;
        }
        if (col < K && t * TILE_WIDTH + ty < N) {
            ds_B[tx][ty] = B[(t * TILE_WIDTH + ty) * K + col];
        } else {
            ds_B[tx][ty] = 0.f;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += ds_A[i][ty] * ds_B[tx][i];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

void init_host_matrix(float* a, float* b, int M, int N, int K);
void compute_host_matrix(float* a, float* b, float* c, int M, int N, int K);
void check_matrix(float* c, float* host_c, int M, int K);

// ./test M N K kernel
int main(int argc, char** argv) {
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    int kernel = atoi(argv[4]);
    printf("M: %d, N: %d, K: %d, kernel: %d\n", M, N, K, kernel);
    assert(argc == 5);

    float *A, *B, *C;
    float* a = (float*)malloc(M * N * sizeof(float));
    float* b = (float*)malloc(N * K * sizeof(float));
    float* c = (float*)malloc(M * K * sizeof(float));
    float* host_c = (float*)malloc(M * K * sizeof(float));

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaMalloc((void**)&A, M * N * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&B, N * K * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&C, M * K * sizeof(float)));

    init_host_matrix(a, b, M, N, K);

    CUDACHECK(cudaMemcpy(A, a, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(B, b, N * K * sizeof(float), cudaMemcpyHostToDevice));

    CUDACHECK(cudaEventRecord(start, 0));
    switch (kernel) {
        case 0: {
            int width = 16;
            dim3 gridSize((M + width - 1) / width, (K + width - 1) / width);
            dim3 blockSize(width, width);
            matmul<<<gridSize, blockSize>>>(A, B, C, M, N, K);
            after_kernel_launch();
            break;
        }
        case 1: {
            dim3 gridSize((M + TILE_WIDTH - 1) / TILE_WIDTH,
                          (K + TILE_WIDTH - 1) / TILE_WIDTH);
            dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
            matmul_share<<<gridSize, blockSize>>>(A, B, C, M, N, K);
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

    CUDACHECK(cudaMemcpy(c, C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
    printf("time: %fms\n", elapsedTime);

    compute_host_matrix(a, b, host_c, M, N, K);
    check_matrix(c, host_c, M, K);

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
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 1.f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = 1.f;
        }
    }
}

void compute_host_matrix(float* a, float* b, float* c, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float value = 0;
            for (int t = 0; t < N; t++) {
                value += a[i * N + t] * b[t * K + j];
            }
            c[i * K + j] = value;
        }
    }
}

void check_matrix(float* c, float* host_c, int M, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            if (c[i * K + j] != host_c[i * K + j]) {
                fprintf(stderr,
                        "check failed: c[%d][%d], host: %f, device: %f\n", i, j,
                        host_c[i * K + j], c[i * K + j]);
                return;
            }
        }
    }
}
