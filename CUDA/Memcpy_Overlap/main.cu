#include <iostream>
#include <memory>
#include <cstring>

#include <cuda_runtime.h>

__global__ void kernel(uint8_t* data) {
    int index = blockIdx.x * 64 + threadIdx.x;
    data[index] += 1;
    data[index] *= data[index];
}

int main(int argc, char* argv[]) {
    const int n = 4;
    cudaStream_t stream[n];
    for(int i = 0; i < n; i++) {
        cudaStreamCreate(&stream[i]);
    }

    int N = 100000000;
    uint8_t* data_h = (uint8_t*)malloc(N);
    memset(data_h, 0, N);
    uint8_t* data = nullptr;
    cudaMalloc(&data, N);
    for(int i = 0; i < n; i++) {
        int offset = i * (N / n);
        cudaMemcpyAsync(&data[offset], &data_h[offset], N / n, cudaMemcpyHostToDevice, stream[i]);
    }

    for(int i = 0; i < n; i++) {
        int offset = i * (N / n);
        kernel<<<(N / n) / 64, 64, 0, stream[i]>>>(&data[offset]);
    }

    for(int i = 0; i < n; i++) {
        int offset = i * (N / n);
        cudaMemcpyAsync(&data_h[offset], &data[offset], N / n, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaDeviceSynchronize();

    for(int i = 0; i < n; i++) {
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(data);
    free(data_h);

    return 0;
}