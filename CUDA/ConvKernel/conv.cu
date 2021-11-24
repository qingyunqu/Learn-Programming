#include "cudnn_conv.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

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

using MMAOp = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm70;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
// Number of pipelines you want to use
constexpr int NumStages = 2;
// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    1,                                                 // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


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
    cutlass::Tensor4DCoord output_size(N, oH, oW, oC);
    cutlass::HostTensor<ElementInputA, LayoutInputA> input(input_size);
    cutlass::HostTensor<ElementInputB, LayoutInputB> filter(filter_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> output(output_size);
    cutlass::HostTensor<ElementOutput, LayoutOutput> output_ref(output_size);
    cutlass::reference::host::TensorFillRandomUniform(input.host_view(), 1, ElementInputA(1), ElementInputA(-1), 0);
    cutlass::reference::host::TensorFillRandomUniform(filter.host_view(), 1, ElementInputB(1), ElementInputB(-1), 0);
    cutlass::reference::host::TensorFill(output.host_view());
    cutlass::reference::host::TensorFill(output_ref.host_view());
    input.sync_device();
    filter.sync_device();
    output.sync_device();
    output_ref.sync_device();

    cudaStream_t stream = nullptr;
    Conv conv(input.device_data(), filter.device_data(), output_ref.device_data(),
              N, iC, iH, iW, oC, kH, kW, oH, oW, strideH, strideW, paddingH, paddingW,
              stream);

    cudaEvent_t start, stop;
    float elapsedTime;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    printf("cudnn:\n");
    for(int i = 0; i <= 10; i++) {
        CUDACHECK(cudaEventRecord(start, stream));
        conv.forward();
        CUDACHECK(cudaEventRecord(stop, stream));
        CUDACHECK(cudaEventSynchronize(stop));
        CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        printf("time: %fms\n", elapsedTime);
    }

    typename cutlass::conv::Conv2dProblemSize problem_size(      
        input_size,
        filter_size,
        {paddingH, paddingH, paddingW, paddingW},
        {strideH, strideW},
        /*dilation*/ {1, 1},
        output_size,
        cutlass::conv::Mode::kCrossCorrelation,
        /*split_k_slices*/ 1);
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        input.device_ref(),
        filter.device_ref(),
        output.device_ref(),
        output.device_ref(),
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
    };
    ImplicitGemm implicit_gemm_op;
    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(implicit_gemm_op.can_implement(arguments));
    CUTLASS_CHECK(implicit_gemm_op.initialize(arguments, workspace.get()));
    printf("cutlass:\n");
    for(int i = 0; i <= 10; i++) {
        CUDACHECK(cudaEventRecord(start, stream));
        CUTLASS_CHECK(implicit_gemm_op());
        CUDACHECK(cudaEventRecord(stop, stream));
        CUDACHECK(cudaEventSynchronize(stop));
        CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        printf("time: %fms\n", elapsedTime);
    }

    output.sync_host();
    output_ref.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(output.host_view(), output_ref.host_view());
    if(!passed) {
        printf("ERROR - results miscompared.\n");
    }

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    return 0;
}