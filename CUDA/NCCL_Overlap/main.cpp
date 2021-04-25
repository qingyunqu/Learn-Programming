#include <iostream>
#include <thread>
#include <memory>

#include <cuda_runtime.h>
#include "comm.h"
#include "cudnn_conv.h"

void run_nccl(int rank, std::unique_ptr<Comm>& comm)
{
    CUDACHECK(cudaSetDevice(rank));
    int N = 1000000;
    void* data = nullptr;
    cudaMalloc(&data, N);
    cudaMemsetAsync(data, 0, N);
    for(int i = 0; i < 200; i++) {
        if(rank == 0) {
            comm->send(data, N, ncclInt8, 1);
        } else if(rank == 1) {
            comm->recv(data, N, ncclInt8, 0);
        }
        //sleep(1);
    }
    comm->syncStream();
    cudaFree(data);
    return NULL;
}

// "./cudnn_conv <rank> <ip> <port>"
int main(int argc, char* argv[]) {

    int nrank = 2;
    int rank = atoi(argv[1]);
    const char* ip = argv[2];
    unsigned short port = (unsigned short)atoi(argv[3]);
    std::unique_ptr<Comm> comm = std::make_unique<Comm>(new Comm(nrank, rank, rank, ip, port));

    std::thread cudnn(run_cudnn, rank);
    std::thread nccl(run_nccl, rank, std::ref(comm));
    cudnn.join();
    nccl.join();
    return 0;
}
