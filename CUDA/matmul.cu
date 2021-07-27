#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include "check.h"

#define TILE_WIDTH 10

//核函数的具体实现
__global__ void matmul(int* M, int* N, int* P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * TILE_WIDTH + tx;
    int Row = by * TILE_WIDTH + ty;

    int Pervalue = 0;

    for (int i = 0; i < width / TILE_WIDTH;
         i++)  //有多少个TILE_WIDTH，每个循环计算一个块的大小
    {
        Mds[ty][tx] = M[Row * width + (i * TILE_WIDTH + tx)];
        Nds[ty][tx] = N[Col + (i * TILE_WIDTH + ty) * width];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)  // TILE_WIDTH相乘
            Pervalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }

    P[Row * width + Col] = Pervalue;
}

int main() {
    const int Nd = 30;
    int Size = Nd * Nd;
    int *M, *N, *P;
    int width = Nd / 3;

    int a[Nd][Nd];
    int b[Nd][Nd];
    int c[Nd][Nd];

    //线程块以及线程的划分
    dim3 gridSize(Nd / width, Nd / width);
    dim3 blockSize(width, width);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //设备内存分配
    cudaMalloc((void**)&M, Size * sizeof(int));
    cudaMalloc((void**)&N, Size * sizeof(int));
    cudaMalloc((void**)&P, Size * sizeof(int));

    //初始化
    for (int i = 0; i < Nd; i++) {
        for (int j = 0; j < Nd; j++) {
            a[i][j] = 2;
            b[i][j] = 3;
        }
    }

    //数据拷贝，主机到设备
    cudaMemcpy(M, a, Size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N, b, Size * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    matmul<<<gridSize, blockSize>>>(M, N, P, Nd);  //调用核函数
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c, P, Size * sizeof(int), cudaMemcpyDeviceToHost);
    printf("time: %dms, c0 = %d\n", elapsedTime, c[0][0]);

    cudaFree(M);
    cudaFree(N);
    cudaFree(P);

    return 0;
}
