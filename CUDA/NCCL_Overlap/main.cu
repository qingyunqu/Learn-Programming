#include <assert.h>
#include <iostream>
#include <memory>
#include <thread>

#include <cuda_runtime.h>
#include <unistd.h>
#include "../check.h"
#include "comm.h"
// #include "cudnn_conv.h"
#include "cutlass_matmul.h"

cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const* A,
                           int lda, float const* B, int ldb, float beta,
                           float* C, int ldc,
                           cudaStream_t s = (cudaStream_t)0) {
    // Define type definition for single-precision CUTLASS GEMM with
    // column-major input matrices and 128x128x8 threadblock tile size (chosen
    // by default).
    //
    // To keep the interface manageable, several helpers are defined for
    // plausible compositions including the following example for
    // single-precision GEMM. Typical values are used as default template
    // arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for
    // more details.
    //
    // To view the full gemm device API interface, see
    // `cutlass/gemm/device/gemm.h`

    using ColumnMajor = cutlass::layout::ColumnMajor;

    using ShapeMMAThreadBlock =
            cutlass::gemm::GemmShape<32, 32, 32>;  // <- threadblock tile M =
                                                    // 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp =
            cutlass::gemm::GemmShape<32, 32, 32>;  // <- warp tile M = 64, N = 64, K = 32
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;  // <- MMA Op tile M =
                                                           // 8, N = 8, K = 4
    using CutlassGemm =
            cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                        ColumnMajor,  // Layout of A matrix
                                        float,        // Data-type of B matrix
                                        ColumnMajor,  // Layout of B matrix
                                        float,        // Data-type of C matrix
                                        ColumnMajor,  // Layout of C matrix
                                        float,
                                        cutlass::arch::OpClassSimt,
                                        cutlass::arch::Sm70,
                                        ShapeMMAThreadBlock,
                                        ShapeMMAWarp,
                                        ShapeMMAOp>;

    cutlass::device_memory::allocation<uint8_t> workspace(0);
    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that
    // are constructible in host code and passed to kernels by value. These may
    // include pointers, strides, scalars, and other arguments needed by Gemm
    // and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy
    // for passing host-constructible arguments to kernels and (2.) minimized
    // initialization overhead on kernel entry.
    //
    CutlassGemm::Arguments args(
            {M, N, K},  // Gemm Problem dimensions
            {A, lda},   // Tensor-ref for source matrix A
            {B, ldb},   // Tensor-ref for source matrix B
            {C, ldc},   // Tensor-ref for source matrix C
            {C, ldc},   // Tensor-ref for destination matrix D (may be different
                        // memory than source C matrix)
            {alpha, beta});  // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //
    cutlass::Status status = gemm_operator.initialize(args, workspace.get(), s);
    CUTLASS_CHECK(status);

    status = gemm_operator(s);
    CUTLASS_CHECK(status);

    // Return success, if no errors were encountered.
    return cudaSuccess;
}

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
                   std::unique_ptr<Comm>& comm1, size_t nbytes) {
    CUDACHECK(cudaSetDevice(rank));
    cudaEvent_t start, start1;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&start1));

    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, nbytes));
    CUDACHECK(cudaMemsetAsync(data, 0, nbytes, comm->getStream()));
    void* data1 = nullptr;
    CUDACHECK(cudaMalloc(&data1, nbytes));
    CUDACHECK(cudaMemsetAsync(data1, 0, nbytes, comm1->getStream()));

    int times = 1000;
    CUDACHECK(cudaEventRecord(start, comm->getStream()));
    CUDACHECK(cudaEventRecord(start1, comm1->getStream()));
    CUDACHECK(cudaStreamWaitEvent(comm->getStream(), start1, 0));
    CUDACHECK(cudaStreamWaitEvent(comm1->getStream(), start1, 0));
    for (int i = 0; i < times + 10; i++) {
        comm->allReduce(data, nbytes, ncclInt8, ncclSum);
        comm1->allReduce(data, nbytes, ncclInt8, ncclSum);
    }
    cudaStreamSynchronize(comm->getStream());
    cudaStreamSynchronize(comm1->getStream());
}

// void run_cudnn(int rank, int argnum = 4) {
//     CUDACHECK(cudaSetDevice(rank));
//     auto arg = arg_list[argnum];
//     int batch_size = arg[0];
//     int C = arg[1];
//     int H = arg[2];
//     int W = arg[3];
//     int kernel_size = arg[4];
//     int K = arg[5];
//     int stride = arg[6];
//     int padding = arg[7];
//     Conv conv(C, K, H, W, batch_size, kernel_size, stride, padding);

//     cudaEvent_t start, stop;
//     CUDACHECK(cudaEventCreate(&start));
//     CUDACHECK(cudaEventCreate(&stop));
//     int times = 100;
//     int effective = 100;
//     for (int i = 0; i < times + 10; i++) {
//         if (i == 10) {
//             CUDACHECK(cudaEventRecord(start, conv.getStream()));
//         }
//         conv.forward();
//         if (i == effective + 9) {
//             CUDACHECK(cudaEventRecord(stop, conv.getStream()));
//         }
//     }
//     CUDACHECK(cudaEventSynchronize(stop));
//     float sum = 0.0;
//     CUDACHECK(cudaEventElapsedTime(&sum, start, stop));

//     std::cout << "single cudnn: (" << arg[0] << "," << arg[1] << "," <<
//     arg[2]
//               << "," << arg[3] << "," << arg[4] << "," << arg[5] << ","
//               << arg[6] << "," << arg[7] << ")"
//               << " average time " << sum / effective << "ms" << std::endl;
//     CUDACHECK(cudaEventDestroy(start));
//     CUDACHECK(cudaEventDestroy(stop));
// }

// void run_cudnn_nccl(int rank, std::unique_ptr<Comm>& comm, size_t nbytes,
//                     int argnum = 4) {
//     CUDACHECK(cudaSetDevice(rank));
//     auto arg = arg_list[argnum];
//     int batch_size = arg[0];
//     int C = arg[1];
//     int H = arg[2];
//     int W = arg[3];
//     int kernel_size = arg[4];
//     int K = arg[5];
//     int stride = arg[6];
//     int padding = arg[7];
//     Conv conv(C, K, H, W, batch_size, kernel_size, stride, padding);

//     // nccl:
//     // int N = 500000000;
//     size_t N = nbytes;
//     void* data = nullptr;
//     CUDACHECK(cudaMalloc(&data, N));
//     CUDACHECK(cudaMemsetAsync(data, 0, N, comm->getStream()));

//     cudaEvent_t start, stop, start1, stop1;
//     CUDACHECK(cudaEventCreate(&start));
//     CUDACHECK(cudaEventCreate(&stop));
//     CUDACHECK(cudaEventCreate(&start1));
//     CUDACHECK(cudaEventCreate(&stop1));
//     CUDACHECK(cudaEventRecord(start, conv.getStream()));
//     CUDACHECK(cudaEventRecord(start1, comm->getStream()));
//     CUDACHECK(cudaStreamWaitEvent(comm->getStream(), start, 0));
//     CUDACHECK(cudaStreamWaitEvent(conv.getStream(), start1, 0));
//     int times = 1000;
//     int effective = 100;
//     for (int i = 0; i < times + 10; i++) {
//         if (i == 10) {
//             CUDACHECK(cudaEventRecord(start, conv.getStream()));
//             CUDACHECK(cudaEventRecord(start1, comm->getStream()));
//         }
//         conv.forward();
//         comm->allReduce(data, N / 4, ncclFloat32, ncclSum);
//         if (i == effective + 9) {
//             CUDACHECK(cudaEventRecord(stop, conv.getStream()));
//             CUDACHECK(cudaEventRecord(stop1, comm->getStream()));
//         }
//     }
//     CUDACHECK(cudaEventSynchronize(stop));
//     CUDACHECK(cudaEventSynchronize(stop1));
//     float sum = 0.0, sum1 = 0.0;
//     CUDACHECK(cudaEventElapsedTime(&sum, start, stop));
//     CUDACHECK(cudaEventElapsedTime(&sum1, start1, stop1));

//     CUDACHECK(cudaEventDestroy(start));
//     CUDACHECK(cudaEventDestroy(stop));
//     CUDACHECK(cudaEventDestroy(start1));
//     CUDACHECK(cudaEventDestroy(stop1));
//     std::cout << "overlap nccl time: " << sum1 / effective << "ms" <<
//     std::endl; std::cout << "overlap cudnn time: " << sum / effective << "ms"
//     << std::endl;
// }

void run_cutlass(int rank, int M = 1024, int N = 1024, int K = 1024) {
    CUDACHECK(cudaSetDevice(rank));
    cudaStream_t stream = (cudaStream_t)0;
    CUDACHECK(cudaStreamCreate(&stream));
    float alpha = 1.0, beta = 0.0;
    int lda = M;
    int ldb = K;
    int ldc = M;
    float* A;
    float* B;
    float* C_cutlass;
    AllocateMatrix(&A, lda, M, K, 0, stream);
    AllocateMatrix(&B, ldb, K, N, 17, stream);
    AllocateMatrix(&C_cutlass, ldc, M, N, 101, stream);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    int times = 100;
    int effective = 100;
    for (int i = 0; i < times + 10; i++) {
        if (i == 10) {
            CUDACHECK(cudaEventRecord(start, stream));
        }
        CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc,
                       stream);
        if (i == effective + 9) {
            CUDACHECK(cudaEventRecord(stop, stream));
        }
    }
    CUDACHECK(cudaEventSynchronize(stop));
    float sum = 0.0;
    CUDACHECK(cudaEventElapsedTime(&sum, start, stop));

    std::cout << "single cutlass: "
              << " average time " << sum / effective << "ms" << std::endl;
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
}

void run_cutlass_nccl(int rank, std::unique_ptr<Comm>& comm, size_t nbytes,
                      int M = 1024, int N = 1024, int K = 1024) {
    CUDACHECK(cudaSetDevice(rank));
    cudaStream_t stream = (cudaStream_t)0;
    CUDACHECK(cudaStreamCreate(&stream));
    float alpha = 1.0, beta = 0.0;
    int lda = M;
    int ldb = K;
    int ldc = M;
    float* A;
    float* B;
    float* C_cutlass;
    AllocateMatrix(&A, lda, M, K, 0, stream);
    AllocateMatrix(&B, ldb, K, N, 17, stream);
    AllocateMatrix(&C_cutlass, ldc, M, N, 101, stream);

    // nccl:
    void* data = nullptr;
    CUDACHECK(cudaMalloc(&data, nbytes));
    CUDACHECK(cudaMemsetAsync(data, 0, nbytes, comm->getStream()));

    cudaEvent_t start, stop, start1, stop1;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventCreate(&start1));
    CUDACHECK(cudaEventCreate(&stop1));
    CUDACHECK(cudaEventRecord(start, stream));
    CUDACHECK(cudaEventRecord(start1, comm->getStream()));
    CUDACHECK(cudaStreamWaitEvent(comm->getStream(), start, 0));
    CUDACHECK(cudaStreamWaitEvent(stream, start1, 0));
    int times = 1000;
    int effective = 100;
    for (int i = 0; i < times + 10; i++) {
        if (i == 10) {
            CUDACHECK(cudaEventRecord(start, stream));
            CUDACHECK(cudaEventRecord(start1, comm->getStream()));
        }
        CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc,
                       stream);
        comm->allReduce(data, nbytes / 4, ncclFloat32, ncclSum);
        if (i == effective + 9) {
            CUDACHECK(cudaEventRecord(stop, stream));
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
    std::cout << "overlap cutlass gemm time: " << sum / effective << "ms"
              << std::endl;
}

// ./test <ip> <port> <rank>
int main(int argc, char* argv[]) {
    //run_cutlass(0);
    //cudaDeviceSynchronize();
    assert(argc == 4);

    int nrank = 2;
    int rank = atoi(argv[3]);
    const char* ip = argv[1];
    unsigned short port = (unsigned short)atoi(argv[2]);

    std::unique_ptr<Comm> comm =
            std::make_unique<Comm>(nrank, rank, rank, ip, port);

    size_t nccl_nbytes = 5000000;
    run_nccl(rank, comm, nccl_nbytes);
    cudaDeviceSynchronize();
    std::cout << std::endl;

    // run_cudnn(rank);
    run_cutlass(rank);
    cudaDeviceSynchronize();
    std::cout << std::endl;

    // run_cudnn_nccl(rank, comm, nccl_nbytes);
    run_cutlass_nccl(rank, comm, nccl_nbytes);
    cudaDeviceSynchronize();
    return 0;
}
