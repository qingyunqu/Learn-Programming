#pragma once

#include <cudnn.h>
#include <cassert>
#include <iostream>
#include "../check.h"

class Conv {
private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    size_t workspace_bytes = 0;
    void* d_workspace{nullptr};
    float* d_input{nullptr};
    float* h_input{nullptr};
    size_t output_bytes;
    float* d_output{nullptr};
    float* h_output{nullptr};
    float* d_filter{nullptr};
    float* h_filter{nullptr};
    const float alpha = 1, beta = 0;
    cudaStream_t m_stream;

public:
    Conv(int N, int iC, int iH, int iW, int oC, int kH, int kW, int strideH,
         int strideW, int paddingH, int paddingW, cudaStream_t stream) {
        this->m_stream = stream;
        auto format = CUDNN_TENSOR_NHWC;
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
        size_t oH = (iH + 2 * paddingH - kH) / strideH + 1;
        size_t oW = (iW + 2 * paddingW - kW) / strideW + 1;
        CUDNNCHECK(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(output_descriptor,
                                              /*format=*/format,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/N,
                                              /*channels=*/oC,
                                              /*image_height=*/oH,
                                              /*image_width=*/oW));
        // filter
        CUDNNCHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
        CUDNNCHECK(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/format,
                                              /*out_channels=*/oC,
                                              /*in_channels=*/iC,
                                              /*kernel_height=*/kH,
                                              /*kernel_width=*/kW));
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
        std::cout << "perfResults status: " << cudnnGetErrorString(perfResults.status)
                  << std::endl;
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

        size_t image_bytes = N * iC * iH * iW * sizeof(float);
        CUDACHECK(cudaMalloc(&d_input, image_bytes));
        // h_input = (float*)malloc(image_bytes);
        // for (int i = 0; i < N * iC * iH * iW; ++i) {
        //     *(h_input + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
        // }
        // CUDACHECK(cudaMemcpyAsync(d_input, h_input, image_bytes,
        //                           cudaMemcpyHostToDevice, m_stream));

        output_bytes = N * oC * oH * oW * sizeof(float);
        CUDACHECK(cudaMalloc(&d_output, output_bytes));
        // CUDACHECK(cudaMemsetAsync(d_output, 0, output_bytes, m_stream));
        // h_output = (float*)malloc(output_bytes);

        size_t filter_bytes = oC * iC * kH * kW * sizeof(float);
        CUDACHECK(cudaMalloc(&d_filter, filter_bytes));
        // h_filter = (float*)malloc(filter_bytes);
        // for (int i = 0; i < oC * iC * kH * kW; ++i) {
        //     *(h_filter + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
        // }
        // CUDACHECK(cudaMemcpyAsync(d_filter, h_filter, filter_bytes,
        //                           cudaMemcpyHostToDevice, m_stream));
    }

    void forward() {
        CUDNNCHECK(cudnnConvolutionForward(
                cudnn, &alpha, input_descriptor, d_input, kernel_descriptor,
                d_filter, convolution_descriptor,
                convolution_algorithm /*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/,
                d_workspace, workspace_bytes, &beta, output_descriptor,
                d_output));
    }

    void copyResult() {
        CUDACHECK(cudaMemcpyAsync(h_output, d_output, output_bytes,
                                  cudaMemcpyDeviceToHost, m_stream));
    }

    cudaStream_t stream() { return this->m_stream; }

    ~Conv() {
        free(h_input);
        free(h_output);
        free(h_filter);

        CUDACHECK(cudaFree(d_input));
        CUDACHECK(cudaFree(d_output));
        CUDACHECK(cudaFree(d_filter));
        CUDACHECK(cudaFree(d_workspace));

        CUDNNCHECK(cudnnDestroyTensorDescriptor(input_descriptor));
        CUDNNCHECK(cudnnDestroyTensorDescriptor(output_descriptor));
        CUDNNCHECK(cudnnDestroyFilterDescriptor(kernel_descriptor));
        CUDNNCHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

        CUDNNCHECK(cudnnDestroy(cudnn));
    }
};
