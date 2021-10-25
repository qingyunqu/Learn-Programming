#include "cudnn_conv.h"

int main() {
    int N = 5;
    int iC = 69;
    int iH = 31;
    int iW = 95;
    int oC = 64;
    int kH = 3;
    int kW = 3;
    int strideH = 1;
    int strideW = 1;
    int paddingH = 1;  // int paddingH0 = 1; int paddingH1 = 1;
    int paddingW = 1;  // int paddingW0 = 1; int paddingW1 = 1;
    //int oH = (iH + 2 * paddingH - kH) / strideH + 1;
    //int oW = (iW + 2 * paddingW - kW) / strideW + 1;

using Ti = float;
using To = float;
    cudaStream_t stream = 0;
    Conv conv(N, iC, iH, iW, oC, kH, kW, strideH, strideW, paddingH, paddingW, stream);

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaEventRecord(start, stream));
    conv.forward();
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("time: %fms\n", elapsedTime);

    CUDACHECK(cudaEventRecord(start, stream));
    conv.forward();
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("time: %fms\n", elapsedTime);

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    return 0;
}