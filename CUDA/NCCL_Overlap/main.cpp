#include <assert.h>
#include <iostream>
#include <memory>
#include <thread>

#include <cuda_runtime.h>
#include <unistd.h>
#include "check.h"
#include "comm.h"
#include "cudnn_conv.h"

const int arg_list[][8] = {
        {256, 256, 14, 14, 3, 512, 1, 1}, {1, 1024, 7, 7, 3, 1024, 1, 1},
        {8, 1024, 7, 7, 3, 1024, 1, 1},   {64, 1024, 7, 7, 3, 1024, 1, 1},
        {256, 1024, 7, 7, 3, 1024, 1, 1}, {1, 1024, 14, 14, 1, 512, 1, 0},
        {1, 256, 28, 28, 3, 512, 1, 1},   {1, 512, 28, 28, 1, 256, 1, 0},
        {1, 128, 56, 56, 3, 256, 1, 1},   {1, 192, 56, 56, 1, 128, 1, 0},
        {1, 64, 112, 112, 3, 192, 1, 1},  {1, 3, 448, 448, 7, 64, 2, 3},
        {1, 3, 448, 448, 7, 64, 2, 3},    {1, 64, 112, 112, 3, 192, 1, 1},
        {1, 192, 56, 56, 1, 128, 1, 0},   {1, 128, 56, 56, 3, 256, 1, 1},
        {1, 256, 56, 56, 1, 256, 1, 0},   {1, 256, 56, 56, 3, 512, 1, 1},
        {1, 512, 28, 28, 1, 256, 1, 0},   {1, 256, 28, 28, 3, 512, 1, 1},
        {1, 512, 28, 28, 1, 512, 1, 0},    // conv15      8
        {1, 512, 28, 28, 3, 1024, 1, 1},   // conv16     9
        {1, 1024, 14, 14, 1, 512, 1, 0},   // conv17    10
        {1, 512, 14, 14, 3, 1024, 1, 1},   // conv18     11
        {1, 1024, 14, 14, 3, 1024, 1, 1},  // conv21   12
        {1, 1024, 14, 14, 3, 1024, 2, 1},  // conv22   13
        {1, 1024, 7, 7, 3, 1024, 1, 1},    // conv23     14
};

void run_nccl(int rank, std::unique_ptr<Comm>& comm, size_t nbytes) {
    CUDACHECK(cudaSetDevice(rank));
    size_t N = nbytes;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, N));
    CUDACHECK(cudaMemsetAsync(data, 0, N, comm->getStream()));
    int times = 100;
    int effective = 100;
    for (int i = 0; i < times + 10; i++) {
        if (i == 10) {
            CUDACHECK(cudaEventRecord(start, comm->getStream()));
        }
        comm->allReduce(data, N / 4, ncclFloat32, ncclSum);
        if (i == effective + 9) {
            CUDACHECK(cudaEventRecord(stop, comm->getStream()));
        }
    }
    CUDACHECK(cudaEventSynchronize(stop));
    float sum = 0.f;
    CUDACHECK(cudaEventElapsedTime(&sum, start, stop));

    CUDACHECK(cudaFree(data));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    std::cout << "single nccl " << N
              << " bytes, average time: " << sum / effective << "ms"
              << std::endl;
}

void run_nccl_nccl(int rank, std::unique_ptr<Comm>& comm,
                   std::unique_ptr<Comm>& comm1) {
    CUDACHECK(cudaSetDevice(rank));
    int N = 500000000;
    cudaEvent_t start, start1;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&start1));

    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, N));
    CUDACHECK(cudaMemsetAsync(data, 0, N, comm->getStream()));
    void* data1 = nullptr;
    CUDACHECK(cudaMalloc(&data1, N));
    CUDACHECK(cudaMemsetAsync(data1, 0, N, comm1->getStream()));

    int times = 1000;
    CUDACHECK(cudaEventRecord(start, comm->getStream()));
    CUDACHECK(cudaEventRecord(start1, comm1->getStream()));
    CUDACHECK(cudaStreamWaitEvent(comm->getStream(), start1, 0));
    CUDACHECK(cudaStreamWaitEvent(comm1->getStream(), start1, 0));
    for (int i = 0; i < times + 10; i++) {
        comm->allReduce(data, N, ncclInt8, ncclSum);
        comm1->allReduce(data, N, ncclInt8, ncclSum);
    }
    cudaStreamSynchronize(comm->getStream());
    cudaStreamSynchronize(comm1->getStream());
}

void run_cudnn(int rank, int argnum = 4) {
    CUDACHECK(cudaSetDevice(rank));
    auto arg = arg_list[argnum];
    int batch_size = arg[0];
    int C = arg[1];
    int H = arg[2];
    int W = arg[3];
    int kernel_size = arg[4];
    int K = arg[5];
    int stride = arg[6];
    int padding = arg[7];
    Conv conv(C, K, H, W, batch_size, kernel_size, stride, padding);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    int times = 100;
    int effective = 100;
    for (int i = 0; i < times + 10; i++) {
        if (i == 10) {
            CUDACHECK(cudaEventRecord(start, conv.getStream()));
        }
        conv.forward();
        if (i == effective + 9) {
            CUDACHECK(cudaEventRecord(stop, conv.getStream()));
        }
    }
    CUDACHECK(cudaEventSynchronize(stop));
    float sum = 0.0;
    CUDACHECK(cudaEventElapsedTime(&sum, start, stop));

    std::cout << "single cudnn: (" << arg[0] << "," << arg[1] << "," << arg[2]
              << "," << arg[3] << "," << arg[4] << "," << arg[5] << ","
              << arg[6] << "," << arg[7] << ")"
              << " average time " << sum / effective << "ms" << std::endl;
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
}

void run_cudnn_nccl(int rank, std::unique_ptr<Comm>& comm, size_t nbytes,
                    int argnum = 4) {
    CUDACHECK(cudaSetDevice(rank));
    auto arg = arg_list[argnum];
    int batch_size = arg[0];
    int C = arg[1];
    int H = arg[2];
    int W = arg[3];
    int kernel_size = arg[4];
    int K = arg[5];
    int stride = arg[6];
    int padding = arg[7];
    Conv conv(C, K, H, W, batch_size, kernel_size, stride, padding);

    // nccl:
    // int N = 500000000;
    size_t N = nbytes;
    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, N));
    CUDACHECK(cudaMemsetAsync(data, 0, N, comm->getStream()));

    cudaEvent_t start, stop, start1, stop1;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventCreate(&start1));
    CUDACHECK(cudaEventCreate(&stop1));
    CUDACHECK(cudaEventRecord(start, conv.getStream()));
    CUDACHECK(cudaEventRecord(start1, comm->getStream()));
    CUDACHECK(cudaStreamWaitEvent(comm->getStream(), start, 0));
    CUDACHECK(cudaStreamWaitEvent(conv.getStream(), start1, 0));
    int times = 1000;
    int effective = 100;
    for (int i = 0; i < times + 10; i++) {
        if (i == 10) {
            CUDACHECK(cudaEventRecord(start, conv.getStream()));
            CUDACHECK(cudaEventRecord(start1, comm->getStream()));
        }
        conv.forward();
        comm->allReduce(data, N / 4, ncclFloat32, ncclSum);
        if (i == effective + 9) {
            CUDACHECK(cudaEventRecord(stop, conv.getStream()));
            CUDACHECK(cudaEventRecord(stop1, comm->getStream()));
        }
    }
    CUDACHECK(cudaEventSynchronize(stop));
    CUDACHECK(cudaEventSynchronize(stop1));
    float sum = 0.0, sum1 = 0.0;
    CUDACHECK(cudaEventElapsedTime(&sum, start, stop));
    CUDACHECK(cudaEventElapsedTime(&sum1, start1, stop1));

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaEventDestroy(start1));
    CUDACHECK(cudaEventDestroy(stop1));
    std::cout << "overlap nccl time: " << sum1 / effective << "ms" << std::endl;
    std::cout << "overlap cudnn time: " << sum / effective << "ms" << std::endl;
}

// ./test <ip> <port> <rank>
int main(int argc, char* argv[]) {
    assert(argc == 4);

    int nrank = 2;
    int rank = atoi(argv[3]);
    const char* ip = argv[1];
    unsigned short port = (unsigned short)atoi(argv[2]);

    std::unique_ptr<Comm> comm =
            std::make_unique<Comm>(nrank, rank, rank, ip, port);

    size_t nccl_nbytes = 100000000;
    run_nccl(rank, comm, nccl_nbytes);
    cudaDeviceSynchronize();
    std::cout << std::endl;
    run_cudnn(rank);
    cudaDeviceSynchronize();
    std::cout << std::endl;
    run_cudnn_nccl(rank, comm, nccl_nbytes);

    cudaDeviceSynchronize();
    return 0;
}
