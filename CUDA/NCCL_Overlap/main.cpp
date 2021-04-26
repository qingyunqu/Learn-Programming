#include <iostream>
#include <memory>
#include <thread>

#include <cuda_runtime.h>
#include <unistd.h>
#include "check.h"
#include "comm.h"
#include "cudnn_conv.h"

void run_nccl(int rank, std::unique_ptr<Comm>& comm) {
    CUDACHECK(cudaSetDevice(rank));
    int N = 500000000;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, N));
    CUDACHECK(cudaMemsetAsync(data, 0, N));
    float sum = 0.f;
    int times = 1000;
    for (int i = 0; i < times + 1; i++) {
        comm->eventRecord(start);
        if (rank == 0) {
            comm->send(data, N, ncclInt8, 1);
        } else if (rank == 1) {
            comm->recv(data, N, ncclInt8, 0);
        }
        comm->eventRecord(stop);
        CUDACHECK(cudaEventSynchronize(stop));
        float elapsed;
        CUDACHECK(cudaEventElapsedTime(&elapsed, start, stop));
        if (i > 0) {
            sum += elapsed;
        }
    }
    comm->syncStream();
    CUDACHECK(cudaFree(data));

    if (rank == 0) {
        std::cout << "send " << N << " bytes, Use time: " << sum / times << "ms"
                  << std::endl;
    } else {
        std::cout << "recv " << N << " bytes, Use time: " << sum / times << "ms"
                  << std::endl;
    }
}

void run_thread(int rank, std::unique_ptr<Comm>& comm) {
    std::thread nccl(run_nccl, rank, std::ref(comm));
    nccl.join();
    std::thread cudnn(run_cudnn, rank);
    cudnn.join();
}

// ./test <ip> <port> <rank>
// ./test
int main(int argc, char* argv[]) {
    if (argc == 1) {
        run_cudnn(0);
        return 0;
    }

    int nrank = 2;
    int rank = atoi(argv[3]);
    const char* ip = argv[1];
    unsigned short port = (unsigned short)atoi(argv[2]);

    std::unique_ptr<Comm> comm =
            std::make_unique<Comm>(nrank, rank, rank, ip, port);

    run_cudnn_nccl(rank, comm);
    //run_thread(rank, comm);
    return 0;
}
