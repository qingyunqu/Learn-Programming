#include "mlir/ExecutionEngine/CRunnerUtils.h"

extern "C" void _mlir_ciface_linalg_fill_f32_viewsxf32(float f, StridedMemRefType<float, 1> *X) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    *(X->data + X->offset + i * X->strides[0]) = f;
}

static void sgemm(const int M, const int N,
                  const int K, const float alpha, const float *A, const int lda,
                  const float *B, const int ldb, const float beta, float *C,
                  const int ldc) {
  for (int m = 0; m < M; ++m) {
    auto *pA = A + m * lda;
    auto *pC = C + m * ldc;
    for (int n = 0; n < N; ++n) {
      float c = pC[n];
      float res = 0.0f;
      for (int k = 0; k < K; ++k) {
        auto *pB = B + k * ldb;
        res += pA[k] * pB[n];
      }
      pC[n] = alpha * c + beta * res;
    }
  }
}

__attribute__((always_inline)) extern "C" void
_mlir_ciface_linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32(
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    StridedMemRefType<float, 2> *C) {
  // printMemRefMetaData(std::cerr, *A);
  sgemm(C->sizes[0], C->sizes[1], A->sizes[1],
        1.0f, A->data + A->offset, A->strides[0], B->data + B->offset,
        B->strides[0], 1.0f, C->data + C->offset, C->strides[0]);
}

__attribute__((always_inline)) extern "C" void
_mlir_ciface_linalg_matmul_view2x4xf32_view4xsxf32_view2xsxf32(
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    StridedMemRefType<float, 2> *C) {
  // printMemRefMetaData(std::cerr, *A);
  sgemm(C->sizes[0], C->sizes[1], A->sizes[1],
        1.0f, A->data + A->offset, A->strides[0], B->data + B->offset,
        B->strides[0], 1.0f, C->data + C->offset, C->strides[0]);
}

__attribute__((always_inline)) extern "C" void
_mlir_ciface_linalg_matmul_view8x16xf32_view16x8xf32_view8x8xf32(
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    StridedMemRefType<float, 2> *C) {
  // printMemRefMetaData(std::cerr, *A);
  sgemm(C->sizes[0], C->sizes[1], A->sizes[1],
        1.0f, A->data + A->offset, A->strides[0], B->data + B->offset,
        B->strides[0], 1.0f, C->data + C->offset, C->strides[0]);
}
