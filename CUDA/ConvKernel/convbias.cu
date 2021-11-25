#include "cudnn_convbias.h"
#include "cudnn_conv.h"
#include "../helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

#include "cutlass/tensor_coord.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using ElementInputA = float;             // Data type of elements in input tensor
using ElementInputB = float;             // Data type of elements in input tensor
using ElementOutput = float;             // Data type of elements in output tensor
using ElementAccumulator = float;        // Data type of accumulator
using ElementComputeEpilogue = float;    // Data type of epilogue computation (alpha, beta)

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

int main() {
    int N = 5;
    int iC = 69; // C
    int iH = 31;
    int iW = 95;
    int oC = 64; // K
    int kH = 3;  // R
    int kW = 3;  // S
    int strideH = 1;
    int strideW = 1;
    int paddingH = 1;  // int paddingH0 = 1; int paddingH1 = 1;
    int paddingW = 1;  // int paddingW0 = 1; int paddingW1 = 1;
    int oH = (iH + 2 * paddingH - kH) / strideH + 1;
    int oW = (iW + 2 * paddingW - kW) / strideW + 1;

    cutlass::Tensor4DCoord input_size(N, iH, iW, iC);
    cutlass::Tensor4DCoord filter_size(oC, kH, kW, iC);
    cutlass::Tensor4DCoord bias_size(1, 1, 1, oC);
    cutlass::Tensor4DCoord output_size(N, oH, oW, oC);
    cutlass::HostTensor<ElementInputA, LayoutInputA> input(input_size);
    cutlass::HostTensor<ElementInputB, LayoutInputB> filter(filter_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> bias(bias_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> output(output_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> output_ref(output_size);
    //cutlass::reference::host::TensorFillRandomUniform(input.host_view(), 1, ElementInputA(1), ElementInputA(-1), 0);
    //cutlass::reference::host::TensorFillRandomUniform(filter.host_view(), 1, ElementInputB(1), ElementInputB(-1), 0);
    //cutlass::reference::host::TensorFillRandomUniform(bias.host_view(), 1, ElementOutput(1), ElementOutput(-1), 0);
    TensorFillRandom<ElementInputA>(input.host_data(), (size_t)N * iH * iW *iC, 1, ElementInputA(1), ElementInputA(-1));
    TensorFillRandom<ElementInputB>(filter.host_data(), (size_t)oC * kH * kW * iC, 1, ElementInputB(1), ElementInputB(-1));
    TensorFillRandom<ElementOutput>(bias.host_data(), (size_t)oC, 1, ElementOutput(1), ElementOutput(-1));
    cutlass::reference::host::TensorFill(output.host_view());
    cutlass::reference::host::TensorFill(output_ref.host_view());
    input.sync_device();
    filter.sync_device();
    bias.sync_device();
    output.sync_device();
    output_ref.sync_device();
    
    cudaStream_t stream = nullptr;
    ConvBias conv_bias(input.device_data(), filter.device_data(), bias.device_data(), output.device_data(),
            N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH, paddingW, stream);
    conv_bias.forward();

    Conv conv(input.device_data(), filter.device_data(), output_ref.device_data(),
            N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH, paddingW, stream);
    conv.forward();

    cudaDeviceSynchronize();
    output.sync_host();
    output_ref.sync_host();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < oH; j++) {
            for (int k = 0; k < oW; k++) {
                for (int l = 0; l < oC; l++) {
                    output_ref.at({i, j, k, l}) = std::max(ElementOutput(0), output_ref.at({i, j, k, l}) + bias.at({0, 0, 0, l}));
                }
            }
        }
    }

    bool passed = TensorEquals<ElementOutput>(output.host_data(), output_ref.host_data(), (size_t)N * oC * oH * oW);
    if (!passed) {
        printf("ERROR - results miscompared.\n");
    }

    return 0; 
}