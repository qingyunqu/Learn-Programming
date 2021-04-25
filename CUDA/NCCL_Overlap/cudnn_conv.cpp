#include "cudnn_conv.h"
#include <cudnn.h>
#include "check.h"

static float conv(int C, int K, int H, int W, int batch_size, int kernel_size,
           int stride, int padding, int times = 1000) {
    /* if (cudaSetDevice(rank) != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) error\n", rank);
    }*/
    srand((unsigned)time(NULL));
    auto format = CUDNN_TENSOR_NHWC;

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/format,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/C,
                                          /*image_height=*/H,
                                          /*image_width=*/W));
    cudnnTensorDescriptor_t output_descriptor;
    size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/format,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/K,
                                          /*image_height=*/H_out,
                                          /*image_width=*/W_out));
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/format,
                                          /*out_channels=*/K,
                                          /*in_channels=*/C,
                                          /*kernel_height=*/kernel_size,
                                          /*kernel_width=*/kernel_size));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(
            cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                            /*pad_height=*/padding,
                                            /*pad_width=*/padding,
                                            /*vertical_stride=*/stride,
                                            /*horizonal_stride=*/stride,
                                            /*dilation_height=*/1,
                                            /*dilation_width=*/1,
                                            /*mode=*/CUDNN_CROSS_CORRELATION,
                                            /*computeType=*/CUDNN_DATA_FLOAT));
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
            cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
            output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            /*memoryLimitInBytes=*/0, &convolution_algorithm));
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
            output_descriptor,
            convolution_algorithm /*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/,
            &workspace_bytes));
    std::cerr << "Workspace size: " << (float(workspace_bytes) / 1048576.0)
              << "MB" << std::endl;
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    size_t image_bytes = batch_size * C * H * W * sizeof(float);

    float* d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    float* h_input{nullptr};
    h_input = (float*)malloc(image_bytes);
    for (int i = 0; i < batch_size * C * H * W; ++i) {
        *(h_input + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
    }
    cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice);

    size_t output_bytes = batch_size * K * H_out * W_out * sizeof(float);

    float* d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);
    float* h_output{nullptr};
    h_output = (float*)malloc(output_bytes);

    size_t filter_bytes = K * C * kernel_size * kernel_size * sizeof(float);

    float* d_filter{nullptr};
    cudaMalloc(&d_filter, filter_bytes);
    float* h_filter{nullptr};
    h_filter = (float*)malloc(filter_bytes);
    for (int i = 0; i < K * C * kernel_size * kernel_size; ++i) {
        *(h_filter + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
    }
    cudaMemcpy(d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice);
    const float alpha = 1, beta = 0;
    // auto beg = (unsigned long long)GetCycleCount();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float sum = 0.0;
    for (int i = 0; i < times + 1; ++i) {
        cudaEventRecord(start, 0);
        checkCUDNN(cudnnConvolutionForward(
                cudnn, &alpha, input_descriptor, d_input, kernel_descriptor,
                d_filter, convolution_descriptor,
                convolution_algorithm /*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/,
                d_workspace, workspace_bytes, &beta, output_descriptor,
                d_output));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        if (i > 0) {
            sum += elapsed;
        }
    }
    // auto end = (unsigned long long)GetCycleCount();
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    free(h_input);
    free(h_filter);
    free(h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);

    return sum;  // float(end - beg);
}

void run_cudnn(int rank) {
    CUDACHECK(cudaSetDevice(rank));
    int arg_lst[][8] = {
            //{256, 256, 14, 14, 3, 512, 1, 1},
            // {1, 1024, 7, 7, 3, 1024, 1, 1},
            // {8, 1024, 7, 7, 3, 1024, 1, 1},
            // {64, 1024, 7, 7, 3, 1024, 1, 1},
            // {256, 1024, 7, 7, 3, 1024, 1, 1},
            // {1, 1024, 14, 14, 1, 512, 1, 0},
            // {1, 256, 28, 28, 3, 512, 1, 1},
            // {1, 512, 28, 28, 1, 256, 1, 0},
            // {1, 128, 56, 56, 3, 256, 1, 1},
            // {1, 192, 56, 56, 1, 128, 1, 0},
            // {1, 64, 112, 112, 3, 192, 1, 1},
            // {1, 3, 448, 448, 7, 64, 2, 3}
            {1, 3, 448, 448, 7, 64, 2, 3},    {1, 64, 112, 112, 3, 192, 1, 1},
            {1, 192, 56, 56, 1, 128, 1, 0},   {1, 128, 56, 56, 3, 256, 1, 1},
            {1, 256, 56, 56, 1, 256, 1, 0},   {1, 256, 56, 56, 3, 512, 1, 1},
            {1, 512, 28, 28, 1, 256, 1, 0},   {1, 256, 28, 28, 3, 512, 1, 1},
            {1, 512, 28, 28, 1, 512, 1, 0},    // conv15      8
            {1, 512, 28, 28, 3, 1024, 1, 1},   // conv16     9
            {1, 1024, 14, 14, 1, 512, 1, 0},   // conv17    10
            {1, 512, 14, 14, 3, 1024, 1, 1},   // conv18     11
            {1, 1024, 14, 14, 3, 1024, 1, 1},  // conv21   12
            {1, 1024, 14, 14, 3, 1024, 2, 1},  // conv22   13
            {1, 1024, 7, 7, 3, 1024, 1, 1},    // conv23     14
    };
    for (int i = 0; i < 15; ++i) {
        int batch_size = arg_lst[i][0];
        int C = arg_lst[i][1];
        int H = arg_lst[i][2];
        int W = arg_lst[i][3];
        int kernel_size = arg_lst[i][4];
        int K = arg_lst[i][5];
        int stride = arg_lst[i][6];
        int padding = arg_lst[i][7];
        int times = 1000;
        std::cout << times << std::endl;
        auto cost = conv(C, K, H, W, batch_size, kernel_size, stride, padding,
                         times);
        std::cout << "(" << batch_size << "," << H << "," << W << "," << C
                  << "," << kernel_size << "," << K << "," << stride << ","
                  << padding << ")"\ << " Use time " << cost / times << "ms"
                  << std::endl;
    }
}
