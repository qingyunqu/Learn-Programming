#include <iostream>
#include <memory>
#include <thread>
#include <assert.h>

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

void run_nccl(int rank, std::unique_ptr<Comm>& comm) {
    CUDACHECK(cudaSetDevice(rank));
    int N = 500000000;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, N));
    CUDACHECK(cudaMemsetAsync(data, 0, N, comm->getStream()));
    float sum = 0.f;
    int times = 1000;
    for (int i = 0; i < times + 1; i++) {
        CUDACHECK(cudaEventRecord(start, comm->getStream()));
        if (rank == 0) {
            comm->send(data, N, ncclInt8, 1);
        } else if (rank == 1) {
            comm->recv(data, N, ncclInt8, 0);
        }
        CUDACHECK(cudaEventRecord(stop, comm->getStream()));
        CUDACHECK(cudaEventSynchronize(stop));
        float elapsed;
        CUDACHECK(cudaEventElapsedTime(&elapsed, start, stop));
        if (i > 0) {
            sum += elapsed;
        }
    }
    cudaStreamSynchronize(comm->getStream());
    CUDACHECK(cudaFree(data));

    std::cout << "nccl " << N << " bytes, Use time: " << sum / times << "ms"
              << std::endl;
}

void run_cudnn(int rank) {
    CUDACHECK(cudaSetDevice(rank));
    auto arg = arg_list[4];
    int batch_size = arg[0];
    int C = arg[1];
    int H = arg[2];
    int W = arg[3];
    int kernel_size = arg[4];
    int K = arg[5];
    int stride = arg[6];
    int padding = arg[7];
    int times = 1000;
    Conv conv(C, K, H, W, batch_size, kernel_size, stride, padding);
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    float sum = 0.0;
    for (int i = 0; i < times + 1; i++) {
        CUDACHECK(cudaEventRecord(start, conv.getStream()));
        conv.forward();
        CUDACHECK(cudaEventRecord(stop, conv.getStream()));
        CUDACHECK(cudaEventSynchronize(stop));
        float elapsed;
        CUDACHECK(cudaEventElapsedTime(&elapsed, start, stop));
        if (i > 0) {
            sum += elapsed;
        }
    }
    cudaStreamSynchronize(conv.getStream());
    std::cout << "(" << batch_size << "," << H << "," << W << "," << C << ","
              << kernel_size << "," << K << "," << stride << "," << padding
              << ")"
              << " Use time " << sum / times << "ms" << std::endl;
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
}

void run_cudnn_nccl(int rank, std::unique_ptr<Comm>& comm) {
    CUDACHECK(cudaSetDevice(rank));
    auto arg = arg_list[4];
    int batch_size = arg[0];
    int C = arg[1];
    int H = arg[2];
    int W = arg[3];
    int kernel_size = arg[4];
    int K = arg[5];
    int stride = arg[6];
    int padding = arg[7];
    int times = 1000;
    Conv conv(C, K, H, W, batch_size, kernel_size, stride, padding);

    // nccl:
    int N = 500000000;
    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, N));
    CUDACHECK(cudaMemsetAsync(data, 0, N, comm->getStream()));

    cudaEvent_t start, stop, start1, stop1;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventCreate(&start1));
    CUDACHECK(cudaEventCreate(&stop1));
    float sum = 0.0, sum1 = 0.0;
    for (int i = 0; i < times + 1; i++) {
        CUDACHECK(cudaEventRecord(start, conv.getStream()));
        CUDACHECK(cudaEventRecord(start1, comm->getStream()));
        CUDACHECK(cudaEventSynchronize(start));
        CUDACHECK(cudaEventSynchronize(start1));
        conv.forward();
        comm->allReduce(data, N, ncclInt8, ncclSum);
        CUDACHECK(cudaEventRecord(stop, conv.getStream()));
        CUDACHECK(cudaEventRecord(stop1, comm->getStream()));
        CUDACHECK(cudaEventSynchronize(stop));
        CUDACHECK(cudaEventSynchronize(stop1));
        float elapsed;
        CUDACHECK(cudaEventElapsedTime(&elapsed, start, stop));
        if (i > 0) {
            sum += elapsed;
        }
        CUDACHECK(cudaEventElapsedTime(&elapsed, start1, stop1));
        if (i > 0) {
            sum1 += elapsed;
        }
    }

    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaEventDestroy(start1));
    CUDACHECK(cudaEventDestroy(stop1));
    std::cout << "cudnn time: " << sum / times << "ms" << std::endl;
    std::cout << "nccl time: " << sum1 / times << "ms" << std::endl;
}

void run_thread(int rank, std::unique_ptr<Comm>& comm) {
    std::thread nccl(run_nccl, rank, std::ref(comm));
    nccl.join();
    std::thread cudnn(run_cudnn, rank);
    cudnn.join();
}

// ./test <ip> <port> <rank>
// ./test <rank>
int main(int argc, char* argv[]) {
    if (argc == 2) {
        run_cudnn(atoi(argv[1]));
        return 0;
    }

    assert(argc == 4);

    int nrank = 2;
    int rank = atoi(argv[3]);
    const char* ip = argv[1];
    unsigned short port = (unsigned short)atoi(argv[2]);

    std::unique_ptr<Comm> comm =
            std::make_unique<Comm>(nrank, rank, rank, ip, port);

    run_cudnn_nccl(rank, comm);
    // run_thread(rank, comm);
    return 0;
}
