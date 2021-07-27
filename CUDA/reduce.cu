#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "check.h"

// ./reduce 10000000 0: 6.23ms
template <typename T>
__global__ void reduce_sum_0(const T* src, T* dst, int N,
                             int ELEMENTS_PER_THREAD) {
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = blockDim.x * i + threadIdx.x;
        if (index < N) {
            atomicAdd(dst, src[index]);
        }
    }
}

// ./reduce 10000000 1: 0.23ms
template <typename T>
__global__ void reduce_sum_1(const T* src, T* dst, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        atomicAdd(dst, src[index]);
    }
}

// ./reduce 10000000 2: 0.11ms
template <typename T>
__global__ void reduce_sum_2(const T* src, T* dst, int N) {
    extern __shared__ T buffer[];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N) {
        buffer[threadIdx.x] = src[index];
    }
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i && index + i < N) {
            buffer[threadIdx.x] += buffer[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(dst, buffer[0]);
    }
}

// ./reduce 10000000 3: 0.23ms
template <typename T>
__global__ void reduce_sum_3(const T* src, T* dst, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(0xffffffff, index < N);
    if (index < N) {
        T val = src[index];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x % 32 == 0) {
            atomicAdd(dst, val);
        }
    }
}

int* init_host_input(int N);
void host_reduce_sum(int* input, int* output, int N);

// reduce size: N -> 1
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <N> <kernel>\n", argv[0]);
        return 0;
    }

    int N = atoi(argv[1]);
    int kernel_id = atoi(argv[2]);

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    int* host_input = init_host_input(N);
    int* host_output = (int*)malloc(sizeof(int) * 1);
    host_reduce_sum(host_input, host_output, N);

    int *device_input = nullptr, *device_output = nullptr;
    CUDACHECK(cudaMalloc(&device_input, sizeof(int) * N));
    CUDACHECK(cudaMalloc(&device_output, sizeof(int) * 1));
    CUDACHECK(cudaMemcpyAsync(device_input, host_input, sizeof(int) * N,
                              cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    int times = 1000;
    cudaEventRecord(start, stream);
    for (int i = 0; i < times; i++) {
        CUDACHECK(cudaMemsetAsync(device_output, 0, sizeof(int) * 1, stream));
        switch (kernel_id) {
            case 0: {
                int threads_per_block = 1024;
                reduce_sum_0<int><<<1, threads_per_block, 0, stream>>>(
                        device_input, device_output, N,
                        (N + threads_per_block - 1) / threads_per_block);
                after_kernel_launch();
                break;
            }
            case 1: {
                int threads_per_block = 128;
                int block = (N + threads_per_block - 1) / threads_per_block;
                reduce_sum_1<int><<<block, threads_per_block, 0, stream>>>(
                        device_input, device_output, N);
                after_kernel_launch();
                break;
            }
            case 2: {
                int threads_per_block = 128;
                int block = (N + threads_per_block - 1) / threads_per_block;
                reduce_sum_2<int><<<block, threads_per_block,
                                    sizeof(int) * threads_per_block, stream>>>(
                        device_input, device_output, N);
                after_kernel_launch();
                break;
            }
            case 3: {
                int threads_per_block = 128;
                int block = (N + threads_per_block - 1) / threads_per_block;
                reduce_sum_3<int><<<block, threads_per_block, 0, stream>>>(
                        device_input, device_output, N);
                after_kernel_launch();
                break;
            }
            default:
                break;
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float time = 0.f;
    cudaEventElapsedTime(&time, start, end);
    printf("N %d kernel %d: average time %.2fms\n", N, kernel_id, time / times);

    int* host_device_output = (int*)malloc(sizeof(int) * 1);
    CUDACHECK(cudaMemcpy(host_device_output, device_output, sizeof(int) * 1,
                         cudaMemcpyDeviceToHost));
    if (host_device_output[0] != host_output[0]) {
        printf("not equal, host: %d <-> device: %d\n", host_output[0],
               host_device_output[0]);
    }
    free(host_device_output);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(device_input);
    cudaFree(device_output);
    cudaStreamDestroy(stream);
    free(host_input);
    free(host_output);
    return 0;
}

int* init_host_input(int N) {
    int* host_input = (int*)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) {
        host_input[i] = rand() % 1000;
    }
    return host_input;
}

void host_reduce_sum(int* input, int* output, int N) {
    *output = 0;
    for (int i = 0; i < N; i++) {
        *output += input[i];
    }
}
