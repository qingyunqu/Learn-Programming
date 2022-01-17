#include "cudnn_convbias.h"
#include "../helper.h"
#include "../check.h"

#include "cutlass/tensor_coord.h"
#include "cutlass/layout/layout.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "cuda_fp16.h"

using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

int main() {
    int N = 8;
    int iC = 128;  // C
    int iH = 224;
    int iW = 224;
    int oC = 128;  // K
    int kH = 3;    // R
    int kW = 3;    // S
    int strideH = 1;
    int strideW = 1;
    int paddingH = 1;  // int paddingH0 = 1; int paddingH1 = 1;
    int paddingW = 1;  // int paddingW0 = 1; int paddingW1 = 1;
    int oH = (iH + 2 * paddingH - kH) / strideH + 1;  // P
    int oW = (iW + 2 * paddingW - kW) / strideW + 1;  // Q

    cutlass::Tensor4DCoord input_size(N, iH, iW, iC);
    cutlass::Tensor4DCoord filter_size(oC, kH, kW, iC);
    cutlass::Tensor4DCoord bias_size(1, 1, 1, oC);
    cutlass::Tensor4DCoord output_size(N, oH, oW, oC);
    cutlass::HostTensor<ElementInputA, LayoutInputA> input(input_size);
    cutlass::HostTensor<ElementInputB, LayoutInputB> filter(filter_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> bias(bias_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> output(output_size);

    TensorFillRandom<ElementInputA>(input.host_data(), (size_t)N * iH * iW *iC, 1, ElementInputA(1), ElementInputA(-1));
    TensorFillRandom<ElementInputB>(filter.host_data(), (size_t)oC * kH * kW * iC, 1, ElementInputB(1), ElementInputB(-1));
    TensorFillRandom<ElementOutput>(bias.host_data(), (size_t)oC, 1, ElementOutput(1), ElementOutput(-1));
    cutlass::reference::host::TensorFill(output.host_view());
    input.sync_device();
    filter.sync_device();
    bias.sync_device();
    output.sync_device();

    cudaStream_t stream = nullptr;
    ConvBias<ElementInputA, ElementInputB, ElementOutput> conv_bias(input.device_data(), filter.device_data(), bias.device_data(), output.device_data(),
            N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH, paddingW, stream);
    // warms up
    for (int i = 0; i < 4; i++) {
        conv_bias.forward();
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    constexpr int runs = 10;
    for (int i = 0; i < runs; i++) {
        conv_bias.forward();
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "cudnn time: " << time / runs << " ms\n";

    return 0;
}