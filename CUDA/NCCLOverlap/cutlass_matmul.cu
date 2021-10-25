#include <iostream>
#include "cutlass_matmul.h"

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(float* matrix, int ldm, int rows,
                                        int columns, int seed = 0) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns) {
        int offset = i + j * ldm;

        // Generate arbitrary elements.
        int const k = 16807;
        int const m = 16;
        float value = float(((offset + seed) * k % m) - m / 2);

        matrix[offset] = value;
    }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float* matrix, int ldm, int rows, int columns,
                             int seed, cudaStream_t s) {
    dim3 block(16, 16);
    dim3 grid((rows + block.x - 1) / block.x,
              (columns + block.y - 1) / block.y);

    InitializeMatrix_kernel<<<grid, block, 0, s>>>(matrix, ldm, rows, columns,
                                                   seed);

    return cudaGetLastError();
}

/// Allocates device memory for a matrix then fills with arbitrary small
/// integers.
cudaError_t AllocateMatrix(float** matrix, int ldm, int rows, int columns,
                           int seed, cudaStream_t s) {
    cudaError_t result;

    size_t sizeof_matrix = sizeof(float) * ldm * columns;

    // Allocate device memory.
    result = cudaMalloc(reinterpret_cast<void**>(matrix), sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: " << cudaGetErrorString(result)
                  << std::endl;
        return result;
    }

    // Clear the allocation.
    result = cudaMemsetAsync(*matrix, 0, sizeof_matrix, s);

    if (result != cudaSuccess) {
        std::cerr << "Failed to clear matrix device memory: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    // Initialize matrix elements to arbitrary small integers.
    result = InitializeMatrix(*matrix, ldm, rows, columns, seed, s);

    if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    return result;
}
