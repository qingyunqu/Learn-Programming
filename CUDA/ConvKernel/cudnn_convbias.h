#pragma once

#include <cudnn.h>
#include <cassert>
#include <iostream>
#include "../check.h"

class ConvBias {
private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnTensorDescriptor_t bias_descriptor;
    cudnnActivationDescriptor_t act_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t workspace_bytes = 0;
    void* d_workspace{nullptr};
    float* d_input{nullptr};
    float* d_filter{nullptr};
    float* d_bias{nullptr};
    float* d_output{nullptr};
    const float alpha1 = 1.f;
    const float alpha2 = 0.f;  // non fuse-z
    cudaStream_t m_stream;

public:
    ConvBias(float* input, float* filter, float* bias, float* output, int N,
             int iC, int iH, int iW, int oC, int kH, int kW, int oH, int oW,
             int strideH, int strideW, int paddingH, int paddingW,
             cudaStream_t stream) {
        this->m_stream = stream;
        auto format = CUDNN_TENSOR_NCHW;
        CUDNNCHECK(cudnnCreate(&cudnn));
        CUDNNCHECK(cudnnSetStream(cudnn, m_stream));
        // input
        CUDNNCHECK(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(input_descriptor,
                                              /*format=*/format,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/N,
                                              /*channels=*/iC,
                                              /*image_height=*/iH,
                                              /*image_width=*/iW));
        // output
        CUDNNCHECK(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(output_descriptor,
                                              /*format=*/format,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/N,
                                              /*channels=*/oC,
                                              /*image_height=*/oH,
                                              /*image_width=*/oW));
        // bias
        CUDNNCHECK(cudnnCreateTensorDescriptor(&bias_descriptor));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(bias_descriptor,
                                              /*format=*/format,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/1,
                                              /*channels=*/oC,
                                              /*image_height=*/1,
                                              /*image_width=*/1));
        // filter
        CUDNNCHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
        CUDNNCHECK(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/format,
                                              /*out_channels=*/oC,
                                              /*in_channels=*/iC,
                                              /*kernel_height=*/kH,
                                              /*kernel_width=*/kW));
        // activation
        CUDNNCHECK(cudnnCreateActivationDescriptor(&act_descriptor));
        CUDNNCHECK(cudnnSetActivationDescriptor(act_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
        // convolution
        CUDNNCHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        CUDNNCHECK(cudnnSetConvolution2dDescriptor(
                convolution_descriptor,
                /*pad_h=*/paddingH,
                /*pad_w=*/paddingW,
                /*u=*/strideH,
                /*v=*/strideW,
                /*dilation_h=*/1,
                /*dilation_w=*/1,
                /*mode=*/CUDNN_CROSS_CORRELATION,
                /*computeType=*/CUDNN_DATA_FLOAT));
        // convolution forward
        int returnedAlgoCount = 0;
        cudnnConvolutionFwdAlgoPerf_t perfResults;
        CUDNNCHECK(cudnnFindConvolutionForwardAlgorithm(
                cudnn, input_descriptor, kernel_descriptor,
                convolution_descriptor, output_descriptor,
                /*requestedAlgoCount=*/1, &returnedAlgoCount, &perfResults));
        assert(returnedAlgoCount == 1);
        convolution_algorithm = perfResults.algo;
        std::cout << "perfResults status: "
                  << cudnnGetErrorString(perfResults.status) << std::endl;
        workspace_bytes = perfResults.memory;
        // workspace size
        // CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(
        //         cudnn, input_descriptor, kernel_descriptor,
        //         convolution_descriptor, output_descriptor,
        //         convolution_algorithm /*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/,
        //         &workspace_bytes));
        std::cerr << "Workspace size: " << (workspace_bytes) << " byte"
                  << std::endl;
        CUDACHECK(cudaMalloc(&d_workspace, workspace_bytes));

        d_input = input;
        d_filter = filter;
        d_bias = bias;
        d_output = output;
    }

    void forward() {
        CUDNNCHECK(cudnnConvolutionBiasActivationForward(
                cudnn, &alpha1, input_descriptor, d_input, kernel_descriptor,
                d_filter, convolution_descriptor,
                convolution_algorithm /*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/,
                d_workspace, workspace_bytes, &alpha2,
                /*zDesc=*/output_descriptor,
                /*z=*/d_output, bias_descriptor, d_bias, act_descriptor,
                output_descriptor, d_output));
    }

    cudaStream_t stream() { return this->m_stream; }

    ~ConvBias() {
        CUDACHECK(cudaFree(d_workspace));

        CUDNNCHECK(cudnnDestroyTensorDescriptor(input_descriptor));
        CUDNNCHECK(cudnnDestroyTensorDescriptor(output_descriptor));
        CUDNNCHECK(cudnnDestroyFilterDescriptor(kernel_descriptor));
        CUDNNCHECK(cudnnDestroyTensorDescriptor(bias_descriptor));
        CUDNNCHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

        CUDNNCHECK(cudnnDestroy(cudnn));
    }
};
