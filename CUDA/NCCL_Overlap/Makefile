CXX := nvcc
CXXFLAGS := -arch=sm_70 -std=c++14 -O2

all: conv

conv: main.cpp comm.h cudnn_conv.h
	$(CXX) $(CXXFLAGS) main.cpp -g -o test \
	-lcudnn -lpthread -lnccl

.phony: clean

clean:
	rm test || echo -n ""