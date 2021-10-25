# CUDA相关代码及实验

### 注：有些代码需要依赖cudnn、nccl、cublas、cutlass等库，注意设置好CPATH,LIBRARY_PATH,LD_LIBRARY_PATH等环境变量

#### ConvKernel
* Convolution实验

#### cutlass_template
* cutlass template for integration

#### MatmulKernel
* Matmul实验

#### MemcpyOverlap
* 单卡上 Host <-> Device 之间memcpy和卡上计算的Overlap实验

#### NCCLOverlap
* NCCL 单机多卡之间的通信和单卡上的计算的Overlap实验

#### ReduceKernel
* Reduce实验
