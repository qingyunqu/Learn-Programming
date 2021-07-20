
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
 
int main()
{
    cudaDeviceProp cudade;
    cudaGetDeviceProperties(&cudade,0);
    std::cout << "GPU型号： " << cudade.name << std::endl;
    std::cout << "每块全局内存存储容量（GB）: " << cudade.totalGlobalMem / 1024 / 1024 << std::endl;
    std::cout << "每块共享内存存储容量（KB）: "<< cudade.sharedMemPerBlock / 1024 << std::endl;
    std::cout << "每块寄存器数量: " << cudade.regsPerBlock << std::endl;
    std::cout << "WarpSize：  " << cudade.warpSize << std::endl;
    std::cout << "最大内存复制步长：  " << cudade.memPitch << std::endl;
    std::cout << "每块最大线程数量：  " << cudade.maxThreadsPerBlock << std::endl;
    std::cout << "线程块三维： " << cudade.maxThreadsDim[0] << "x" << cudade.maxThreadsDim[1] << "x" << cudade.maxThreadsDim[2] << std::endl;
    std::cout << "线程格三维: " << cudade.maxGridSize[0] << "x" << cudade.maxGridSize[1] << "x" << cudade.maxGridSize[2] << std::endl;
    std::cout << "计算核心时钟频率（kHz）：  " << cudade.clockRate << std::endl;
    std::cout << "常量存储容量：  " << cudade.totalConstMem << std::endl;
    std::cout << "次计算能力（小数点后的值）： " << cudade.minor << std::endl;
    std::cout << "纹理对齐要求: " << cudade.textureAlignment << std::endl;
    std::cout << "绑定到等步长内存的纹理满足的要求: " << cudade.texturePitchAlignment << std::endl;
    std::cout << "GPU是否支持并发内存复制和kernel执行: " << cudade.deviceOverlap << std::endl;
    std::cout << "SMX数量：  " << cudade.multiProcessorCount << std::endl;
    std::cout << "是否有运行时限制：  " << cudade.kernelExecTimeoutEnabled << std::endl;
    std::cout << "设备是否集成（否则独立）：  " << cudade.integrated << std::endl;
    std::cout << "可否对主机内存进行映射： " << cudade.canMapHostMemory << std::endl;
    std::cout << "计算模式: " << cudade.computeMode << std::endl;
    std::cout << "最大1D纹理尺寸：  " << cudade.maxTexture1D << std::endl;
    std::cout << "线性内存相关的最大1D纹理尺寸：  " << cudade.maxTexture1DLinear << std::endl;
    std::cout << "最大2D纹理维度：  " << cudade.maxTexture2D << std::endl;
    std::cout << "最大2D纹理维度（width,height,pitch）： " << cudade.maxTexture2DLinear << std::endl;
    std::cout << "纹理聚集时的最大纹理维度: " << cudade.maxTexture2DGather << std::endl;
    std::cout << "最大3D纹理维度: " << cudade.maxTexture3D << std::endl;
    std::cout << "最大立方图纹理维度: " << cudade.maxTextureCubemap << std::endl;
    std::cout << "最大1D分层纹理维度：  " << cudade.maxTexture1DLayered << std::endl;
    std::cout << "最大2D分层纹理维度：  " << cudade.maxTexture2DLayered << std::endl;
    std::cout << "最大立方图分层纹理维度：  " << cudade.maxTextureCubemapLayered << std::endl;
    std::cout << "最大1D表面尺寸： " << cudade.maxSurface1D << std::endl;
    std::cout << "主计算能力（小数点前的值）：  " << cudade.major << std::endl;
    std::cout << "最大2D表面维度: " << cudade.maxSurface2D << std::endl;
    std::cout << "最大3D表面维度：  " << cudade.maxSurface3D << std::endl;
    std::cout << "最大1D分层表面维度：  " << cudade.maxSurface1DLayered << std::endl;
    std::cout << "最大2D分层表面维度：  " << cudade.maxSurface2DLayered << std::endl;
    std::cout << "最大立方图表面维度： " << cudade.maxSurfaceCubemap << std::endl;
    std::cout << "表面对齐要求: " << cudade.surfaceAlignment << std::endl;
    std::cout << "设备能并发的kernel数量: " << cudade.concurrentKernels << std::endl;
    std::cout << "是否打开ECC校验: " << cudade.ECCEnabled << std::endl;
    std::cout << "PCI总线ID：  " << cudade.pciBusID << std::endl;
    std::cout << "PCI设备ID：  " << cudade.pciDeviceID << std::endl;
    std::cout << "PCI域ID：  " << cudade.pciDomainID << std::endl;
    std::cout << "是否支持TCC（Tesla集群）： " << cudade.tccDriver << std::endl;
    std::cout << "异步引擎数量: " << cudade.asyncEngineCount << std::endl;
    std::cout << "主机和设备共享同一地址空间：  " << cudade.unifiedAddressing << std::endl;
    std::cout << "存储时钟频率：  " << cudade.memoryClockRate << std::endl;
    std::cout << "Global memory 总线带宽：  " << cudade.memoryBusWidth << std::endl;
    std::cout << "L2 Cache尺寸（B)：  " << cudade.l2CacheSize << std::endl;
    std::cout << "每个SMX驻留的最大线程数量：  " << cudade.maxThreadsDim << std::endl;
}
