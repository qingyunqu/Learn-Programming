#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float* matrix, int ldm, int rows, int columns,
                             int seed = 0, cudaStream_t s = (cudaStream_t)0);

/// Allocates device memory for a matrix then fills with arbitrary small
/// integers.
cudaError_t AllocateMatrix(float** matrix, int ldm, int rows, int columns,
    int seed = 0, cudaStream_t s = (cudaStream_t)0);
