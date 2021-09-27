# Tiling on CPU
#### No Tiling
* `mlir-opt tile-test.mlir --convert-linalg-to-std -convert-linalg-to-llvm --convert-std-to-llvm | mlir-translate --mlir-to-llvmir > tile-test.ll`
#### Tiling
* `mlir-opt tile-test.mlir -linalg-tile="linalg-tile-sizes=8,8,16" -mlir-disable-threading=true --convert-linalg-to-std  --convert-scf-to-std --lower-affine --convert-linalg-to-llvm --convert-std-to-llvm | mlir-translate --mlir-to-llvmir > tile-test-1.ll`
#### Compile
* `clang++ -O3 -emit-llvm -S cblas.cpp -std=c++14 -I /root/share/llvm-project/mlir/include -o cblas.ll`
* `clang++ -O3 main.cpp tile-test.ll cblas.ll -o tile-test`