CXX := nvcc
CXXFLAGS := -arch=sm_70 -std=c++14 -O2

all: conv

conv: cudnn_conv.cpp comm.cpp main.cpp
	$(CXX) $(CXXFLAGS)  cudnn_conv.cpp comm.cpp main.cpp -o test \
	-lcudnn -lpthread -lnccl

.phony: clean

clean:
	rm test || echo -n ""