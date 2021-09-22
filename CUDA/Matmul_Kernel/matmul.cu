#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include "../check.h"

__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N) {
        float value = 0.f;
        for(int i = 0; i < N; i++) {
            value += A[row * N + i] * B[col + i * N];
        }
        C[row * N + col] = value;
        //printf("row: %d, col: %d, value: %f\n", row, col, value);
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

int main() {
    const int Nd = 256;
    int width = 16;
    int Size = Nd * Nd;
    float *A, *B, *C;

    float a[Nd][Nd];
    float b[Nd][Nd];
    float c[Nd][Nd];

    dim3 gridSize(Nd / width, Nd / width);
    dim3 blockSize(width, width);

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaMalloc((void**)&A, Size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&B, Size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)&C, Size * sizeof(float)));

    for (int i = 0; i < Nd; i++) {
        for (int j = 0; j < Nd; j++) {
            a[i][j] = 2.f;
            b[i][j] = 3.f;
        }
    }

    CUDACHECK(cudaMemcpy(A, a, Size * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(B, b, Size * sizeof(float), cudaMemcpyHostToDevice));

    CUDACHECK(cudaEventRecord(start, 0));
    matmul<<<gridSize, blockSize>>>(A, B, C, Nd);
    after_kernel_launch();
    CUDACHECK(cudaEventRecord(stop, 0));
    CUDACHECK(cudaEventSynchronize(stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    CUDACHECK(cudaMemcpy(c, C, Size * sizeof(float), cudaMemcpyDeviceToHost));
    printf("time: %fms, c0 = %f\n", elapsedTime, c[0][0]);

    CUDACHECK(cudaFree(A));
    CUDACHECK(cudaFree(B));
    CUDACHECK(cudaFree(C));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return 0;
}
