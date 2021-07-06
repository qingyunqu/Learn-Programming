#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

template <typename T>
__global__ void reduce_sum_version_0(T* src, T* dst, int M, int N) {
    T* src_ptr = src + blockIdx.x * N;
    dst[blockIdx.x] = static_cast<T>(0);
    for (int i = 0; i < N; i++) {
        dst[blockIdx.x] += src_ptr[i];
    }
}

int* init_host_input(int M, int N) {
    int* host_input = (int*)malloc(sizeof(int) * M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            host_input[i * N + j] = rand();
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

    int* host_input = init_host_input(M, N);
    int* host_output = (int*)malloc(sizeof(int) * M);
    host_reduce_sum(host_input, host_output, M, N);

    int *device_input = nullptr, device_output = nullptr;
    cudaMalloc(&device_input, sizeof(int) * M * N);
    cudaMalloc(&device_output, sizeof(int) * M);
    cudaMemcpy(device_input, host_input, sizeof(int) * M * N,
               cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    int times = 1000;
    cudaEventRecord(start, (cudaStream_t)0);
    for (int i = 0; i < time; i++) {
        switch (kernel_id) {
            case 0:
                reduce_sum_version_0<int>
                        <<<M, 1>>>(device_input, device_output, M, N);
                break;
            default:
                break;
        }
    }
    cudaEventRecord(end, (cudaStream_t)0);
    cudaEventSynchronize(end);
    float time = 0.f;
    cudaEventElapsedTime(&time, start, end);
    printf("M %d N %d kernel %d: average time %.2fms\n", M, N, kernel_id,
           time / times);

    int* host_device_output = (int*)malloc(sizeof(int) * M);
    cudaMemcpy(host_device_output, device_output, sizeof(int) * M,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        if (host_device_output[i] != host_output[i]) {
            printf("index: %d, not equal %d <-> %d", host_output[i],
                   host_device_output[i]);
            break;
        }
    }
    free(host_device_output);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(device_input);
    cudaFree(device_output);
    free(host_input);
    free(host_output);
    return 0;
}