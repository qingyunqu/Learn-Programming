CXX := nvcc
CXXFLAGS := -arch=sm_70 -std=c++14 -O3
HEADPATH := -I/root/share/cutlass/include -I/root/share/cutlass/tools/util/include

all: target

target: matmul.cu
	$(CXX) $(CXXFLAGS) $< -o test -lcublas $(HEADPATH)

.phony: clean

clean:
	rm test || echo -n ""
