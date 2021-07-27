#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "check.h"

template <typename T>
__global__ void reduce_sum_naive(const T* src, T* dst, int M, int N) {
    const T* src_ptr = src + blockIdx.x * N;
    dst[blockIdx.x] = static_cast<T>(0);
    for (int i = 0; i < N; i++) {
        dst[blockIdx.x] += src_ptr[i];
    }
}

template <typename T>
__global__ void reduce_sum_version_0(const T* src, T* dst, int M, int N) {
    // every block has 1024 threads
    const int ELEMENTS_PER_THREAD = (N + blockDim.x - 1) / blockDim.x;
    const T* src_ptr = src + blockIdx.x * N;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = threadIdx.x + i * blockDim.x;
        if (index < N) {
            atomicAdd(&dst[blockIdx.x], src_ptr[index]);
        }
    }
}

template <typename T>
__global__ void reduce_sum_version_1(const T* src, T* dst, int M, int N) {
    extern __shared__ T buffer[];
    *buffer = 0;
    __syncthreads();
    // every block has 1024 threads
    const int ELEMENTS_PER_THREAD = (N + blockDim.x - 1) / blockDim.x;
    const T* src_ptr = src + blockIdx.x * N;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = threadIdx.x + i * blockDim.x;
        if (index < N) {
            atomicAdd(buffer, src_ptr[index]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        dst[blockIdx.x] = *buffer;
    }
}

#define BUFFER_SIZE 32
template <typename T>
__global__ void reduce_sum_version_2(const T* src, T* dst, int M, int N) {
    extern __shared__ T buffer[];
    if (threadIdx.x == 0) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            buffer[i] = 0;
        }
    }
    __syncthreads();
    const int ELEMENTS_PER_THREAD = (N + blockDim.x - 1) / blockDim.x;
    const T* src_ptr = src + blockIdx.x * N;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = threadIdx.x + i * blockDim.x;
        if (index < N) {
            int buffer_index = index % BUFFER_SIZE;
            atomicAdd(&buffer[buffer_index], src_ptr[index]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            dst[blockIdx.x] += buffer[i];
        }
    }
}

template <typename T>
__global__ void reduce_sum_version_3(const T* src, T* dst, int M, int N) {
    extern __shared__ T buffer[];
    const int ELEMENTS_PER_THREAD = (N + blockDim.x - 1) / blockDim.x;
    const T* src_ptr = src + blockIdx.x * N;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        if (i == 0) {
            buffer[threadIdx.x] = 0;
        }
        int index = threadIdx.x + i * blockDim.x;
        if (index < N) {
            buffer[threadIdx.x] += src_ptr[index];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            dst[blockIdx.x] += buffer[i];
        }
    }
}

#define WARP_SIZE 32
template <typename T>
__global__ void reduce_sum_version_4(const T* src, T* dst, int M, int N) {
    const int ELEMENTS_PER_THREAD = (N + blockDim.x - 1) / blockDim.x;
    const T* src_ptr = src + blockIdx.x * N;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int index = threadIdx.x + i * blockDim.x;
        unsigned mask = __ballot_sync(0xffffffff, index < N);
        if (index < N) {
            T val = src_ptr[index];
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(mask, val, offset);
            }
            if (index % WARP_SIZE == 0) {
                atomicAdd(&dst[blockIdx.x], val);
            }
        }
    }
}

int* init_host_input(int M, int N) {
    int* host_input = (int*)malloc(sizeof(int) * M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            host_input[i * N + j] = rand() % 1000;
        }
    }
    return host_input;
}

void host_reduce_sum(int* input, int* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        output[i] = 0;
        for (int j = 0; j < N; j++) {
            output[i] += input[i * N + j];
        }
    }
}

// reduce size: M * N -> M
int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: ./%s <M> <N> <kernel>\n", argv[0]);
        return 0;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int kernel_id = atoi(argv[3]);

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    int* host_input = init_host_input(M, N);
    int* host_output = (int*)malloc(sizeof(int) * M);
    host_reduce_sum(host_input, host_output, M, N);

    int *device_input = nullptr, *device_output = nullptr;
    int* device_buffer = nullptr;
    CUDACHECK(cudaMalloc(&device_input, sizeof(int) * M * N));
    CUDACHECK(cudaMalloc(&device_output, sizeof(int) * M));
    CUDACHECK(cudaMalloc(&device_buffer, sizeof(int) * M * N));
    CUDACHECK(cudaMemcpyAsync(device_input, host_input, sizeof(int) * M * N,
                              cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    int times = 1000;
    cudaEventRecord(start, stream);
    for (int i = 0; i < times; i++) {
        CUDACHECK(cudaMemsetAsync(device_output, 0, sizeof(int) * M, stream));
        switch (kernel_id) {
            case -1: {
                reduce_sum_naive<int>
                        <<<M, 1>>>(device_input, device_output, M, N);
                after_kernel_launch();
                break;
            }
            case 0: {
#define MAX_BLOCK_THREADS 1024
                int threads = MAX_BLOCK_THREADS;
                reduce_sum_version_0<int><<<M, threads, 0, stream>>>(
                        device_input, device_output, M, N);
                after_kernel_launch();
                break;
            }
            case 1: {
                int threads = MAX_BLOCK_THREADS;
                reduce_sum_version_1<int><<<M, threads, sizeof(int), stream>>>(
                        device_input, device_output, M, N);
                after_kernel_launch();
                break;
            }
            case 2: {
                int threads = MAX_BLOCK_THREADS;
                reduce_sum_version_2<int>
                        <<<M, threads, sizeof(int) * BUFFER_SIZE, stream>>>(
                                device_input, device_output, M, N);
                after_kernel_launch();
                break;
            }
            case 3: {
                int threads = MAX_BLOCK_THREADS;
                reduce_sum_version_3<int>
                        <<<M, threads, sizeof(int) * MAX_BLOCK_THREADS,
                           stream>>>(device_input, device_output, M, N);
                after_kernel_launch();
                break;
            }
            case 4: {
                int threads = MAX_BLOCK_THREADS;
                reduce_sum_version_4<int><<<M, threads, 0, stream>>>(
                        device_input, device_output, M, N);
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
    printf("M %d N %d kernel %d: average time %.2fms\n", M, N, kernel_id,
           time / times);

    int* host_device_output = (int*)malloc(sizeof(int) * M);
    CUDACHECK(cudaMemcpy(host_device_output, device_output, sizeof(int) * M,
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; i++) {
        if (host_device_output[i] != host_output[i]) {
            printf("index: %d, not equal %d <-> %d\n", i, host_output[i],
                   host_device_output[i]);
            break;
        }
    }
    free(host_device_output);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_buffer);
    cudaStreamDestroy(stream);
    free(host_input);
    free(host_output);
    return 0;
}
