CXX := nvcc
CXXFLAGS := -arch=sm_70 -std=c++14 -O2

all: conv

conv: main.cu comm.h cudnn_conv.h cutlass_matmul.cu
	$(CXX) $(CXXFLAGS) main.cu cutlass_matmul.cu -o test \
	-lcudnn -lpthread -lnccl

.phony: clean

clean:
	rm test || echo -n ""
