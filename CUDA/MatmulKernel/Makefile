CXX := nvcc
CXXFLAGS := -arch=sm_70 -std=c++14 -O3
HEADPATH := -I/root/share/cutlass/include -I/root/share/cutlass/tools/util/include

all: matmul_all matmulbias

matmul_all: matmul_all.cu
	$(CXX) $(CXXFLAGS) $< -o $@ -lcublas $(HEADPATH)

matmulbias: matmulbias.cu
	$(CXX) $(CXXFLAGS) $< -o $@ $(HEADPATH)

.phony: clean

clean:
	rm test || echo -n ""
