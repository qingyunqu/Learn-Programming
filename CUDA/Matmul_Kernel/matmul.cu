#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include "../check.h"

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

// #define TILE_WIDTH 10
// __global__ void matmul(int* M, int* N, int* P, int width) {
//     __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     int Col = bx * TILE_WIDTH + tx;
//     int Row = by * TILE_WIDTH + ty;

//     int Pervalue = 0;

//     for (int i = 0; i < width / TILE_WIDTH;
//          i++)  //有多少个TILE_WIDTH，每个循环计算一个块的大小
//     {
//         Mds[ty][tx] = M[Row * width + (i * TILE_WIDTH + tx)];
//         Nds[ty][tx] = N[Col + (i * TILE_WIDTH + ty) * width];
//         __syncthreads();

//         for (int k = 0; k < TILE_WIDTH; k++)  // TILE_WIDTH相乘
//             Pervalue += Mds[ty][k] * Nds[k][tx];
//         __syncthreads();
//     }

//     P[Row * width + Col] = Pervalue;
// }

void init_host_matrix(float* a, float* b, int M, int N, int K) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            a[i * N + j] = 1.f;
        }
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
            b[i * K + j] = 1.f;
        }
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    int width = 16;
    float *A, *B, *C;

    float* a = (float*)malloc(M * N * sizeof(float));
    float* b = (float*)malloc(N * K * sizeof(float));
    float* c = (float*)malloc(M * K * sizeof(float));

    dim3 gridSize((M + width - 1) / width, (K + width - 1) / width);
    dim3 blockSize(width, width);

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
    matmul<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    after_kernel_launch();
    CUDACHECK(cudaEventRecord(stop, 0));
    CUDACHECK(cudaEventSynchronize(stop));
    //CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    CUDACHECK(cudaMemcpy(c, C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
    printf("time: %fms, c0 = %f\n", elapsedTime, c[0]);

    free(a);
    free(b);
    free(c);
    CUDACHECK(cudaFree(A));
    CUDACHECK(cudaFree(B));
    CUDACHECK(cudaFree(C));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return 0;
}
