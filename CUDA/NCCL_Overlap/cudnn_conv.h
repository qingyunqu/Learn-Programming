#pragma once

#include <cudnn.h>
#include <memory>
#include "check.h"

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
    bool m_internl_stream = false;

public:
    Conv(int C, int K, int H, int W, int batch_size, int kernel_size,
         int stride, int padding, cudaStream_t stream) {
        this->m_stream = stream;
        srand((unsigned)time(NULL));
        auto format = CUDNN_TENSOR_NHWC;
        CUDNNCHECK(cudnnCreate(&cudnn));
        CUDNNCHECK(cudnnSetStream(cudnn, m_stream));
        // input
        CUDNNCHECK(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(input_descriptor,
                                              /*format=*/format,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/batch_size,
                                              /*channels=*/C,
                                              /*image_height=*/H,
                                              /*image_width=*/W));
        // output
        size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
        size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
        CUDNNCHECK(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(output_descriptor,
                                              /*format=*/format,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/batch_size,
                                              /*channels=*/K,
                                              /*image_height=*/H_out,
                                              /*image_width=*/W_out));
        // filter
        CUDNNCHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
        CUDNNCHECK(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/format,
                                              /*out_channels=*/K,
                                              /*in_channels=*/C,
                                              /*kernel_height=*/kernel_size,
                                              /*kernel_width=*/kernel_size));
        // convolution
        CUDNNCHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        CUDNNCHECK(cudnnSetConvolution2dDescriptor(
                convolution_descriptor,
                /*pad_height=*/padding,
                /*pad_width=*/padding,
                /*vertical_stride=*/stride,
                /*horizonal_stride=*/stride,
                /*dilation_height=*/1,
                /*dilation_width=*/1,
                /*mode=*/CUDNN_CROSS_CORRELATION,
                /*computeType=*/CUDNN_DATA_FLOAT));
        // convolution forward
        CUDNNCHECK(cudnnGetConvolutionForwardAlgorithm(
                cudnn, input_descriptor, kernel_descriptor,
                convolution_descriptor, output_descriptor,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                /*memoryLimitInBytes=*/0, &convolution_algorithm));
        // workspace size
        CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(
                cudnn, input_descriptor, kernel_descriptor,
                convolution_descriptor, output_descriptor,
                convolution_algorithm /*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/,
                &workspace_bytes));
        std::cerr << "Workspace size: " << (float(workspace_bytes) / 1048576.0)
                  << "MB" << std::endl;
        CUDACHECK(cudaMalloc(&d_workspace, workspace_bytes));

        size_t image_bytes = batch_size * C * H * W * sizeof(float);
        CUDACHECK(cudaMalloc(&d_input, image_bytes));
        h_input = (float*)malloc(image_bytes);
        for (int i = 0; i < batch_size * C * H * W; ++i) {
            *(h_input + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
        }
        CUDACHECK(cudaMemcpyAsync(d_input, h_input, image_bytes,
                                  cudaMemcpyHostToDevice, m_stream));

        output_bytes = batch_size * K * H_out * W_out * sizeof(float);
        CUDACHECK(cudaMalloc(&d_output, output_bytes));
        CUDACHECK(cudaMemsetAsync(d_output, 0, output_bytes, m_stream));
        h_output = (float*)malloc(output_bytes);

        size_t filter_bytes = K * C * kernel_size * kernel_size * sizeof(float);
        CUDACHECK(cudaMalloc(&d_filter, filter_bytes));
        h_filter = (float*)malloc(filter_bytes);
        for (int i = 0; i < K * C * kernel_size * kernel_size; ++i) {
            *(h_filter + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
        }
        CUDACHECK(cudaMemcpyAsync(d_filter, h_filter, filter_bytes,
                                  cudaMemcpyHostToDevice, m_stream));
    }

    Conv(int C, int K, int H, int W, int batch_size, int kernel_size,
         int stride, int padding) {
        CUDACHECK(cudaStreamCreate(&this->m_stream));
        m_internl_stream = true;
        new (this) Conv(C, K, H, W, batch_size, kernel_size, stride, padding,
                        this->m_stream);
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

    cudaStream_t getStream() { return this->m_stream; }

    ~Conv() {
        free(h_input);
        free(h_filter);
        free(h_output);

        CUDACHECK(cudaFree(d_input));
        CUDACHECK(cudaFree(d_output));
        CUDACHECK(cudaFree(d_filter));
        CUDACHECK(cudaFree(d_workspace));

        CUDNNCHECK(cudnnDestroyTensorDescriptor(input_descriptor));
        CUDNNCHECK(cudnnDestroyTensorDescriptor(output_descriptor));
        CUDNNCHECK(cudnnDestroyFilterDescriptor(kernel_descriptor));
        CUDNNCHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

        CUDNNCHECK(cudnnDestroy(cudnn));

        if (m_internl_stream) {
            CUDACHECK(cudaStreamDestroy(this->m_stream));
        }
    }
};
