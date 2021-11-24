CXX := nvcc
CXXFLAGS := -arch=sm_70 -std=c++14 -O3
HEADPATH := -I/root/share/cutlass/include -I/root/share/cutlass/tools/util/include

all: conv convbias

conv: conv.cu
	$(CXX) $(CXXFLAGS) $< -o $@ -lcudnn $(HEADPATH)

convbias: convbias.cu
	$(CXX) $(CXXFLAGS) $< -o $@ -lcudnn $(HEADPATH)

.phony: clean

clean:
	rm conv convbias || echo -n ""
