## NCCL Overlap with Cutlass Gemm

* Compile: make
* Run:
  * Open a Terminal: ./test 127.0.0.1 10000 0
  * Open another Terminal: ./test 127.0.0.1 10000 1
* Modify `ShapeMMAThreadBlock`(main.cu:124) and `ShapeMMAWaro`(main.cu:128) to get different blocks shape of gemm.