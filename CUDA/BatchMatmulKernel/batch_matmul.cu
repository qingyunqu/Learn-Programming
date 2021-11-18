/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

 #include <iostream>
 #include <vector>
 #include "../check.h"
 
 #include "cutlass/cutlass.h"
 #include "cutlass/layout/matrix.h"
 #include "cutlass/gemm/device/gemm_batched.h"
 
 #pragma warning( disable : 4503)
 
 /*
 This example demonstrates how to use cutlass to compute a batched strided gemm.
 In this example, both A and B matrix are non-transpose and column major matrix
 batched_C = batched_A x batched_B
 As an example, matrix C can be seen as
 -----------------------------------------------------------
 (0,0,0) | (0,0,1) | (0,0,2) | (1,0,0) | (1,0,1) | (1,0,2) |
 -----------------------------------------------------------
 (0,1,0) | (0,1,1) | (0,1,2) | (1,1,0) | (1,1,1) | (1,1,2) |
 -----------------------------------------------------------
 (0,2,0) | (0,2,1) | (0,2,2) | (1,2,0) | (1,2,1) | (1,2,2) |
 -----------------------------------------------------------
 (0,3,0) | (0,3,1) | (0,3,2) | (1,3,0) | (1,3,1) | (1,3,2) |
 -----------------------------------------------------------
 (0,4,0) | (0,4,1) | (0,4,2) | (1,4,0) | (1,4,1) | (1,4,2) |
 -----------------------------------------------------------
 (0,5,0) | (0,5,1) | (0,5,2) | (1,5,0) | (1,5,1) | (1,5,2) |
 -----------------------------------------------------------
            batch 0          |           batch 1
 where we denote each element with (batch_idx, row_idx, column_idx)
 In this example, batch size is 2, M is 6 and N is 3
 The stride (batch_stride_C) between the first element of two batches is ldc * n
 
 matrix A can be seen as
 ---------------------------------------
 (0,0,0) | (0,0,1) | (1,0,0) | (1,0,1) |
 ---------------------------------------
 (0,1,0) | (0,1,1) | (1,1,0) | (1,1,1) |
 ---------------------------------------
 (0,2,0) | (0,2,1) | (1,2,0) | (1,2,1) |
 ---------------------------------------
 (0,3,0) | (0,3,1) | (1,3,0) | (1,3,1) |
 ---------------------------------------
 (0,4,0) | (0,4,1) | (1,4,0) | (1,4,1) |
 ---------------------------------------
 (0,5,0) | (0,5,1) | (1,5,0) | (1,5,1) |
 ---------------------------------------
      batch 0      |      batch 1
 , where batch size is 2, M is 6 and K is 2
 The stride (batch_stride_B) between the first element of two batches is lda * k
 
 matrix B can be seen as
 -----------------------------
 (0,0,0) | (0,0,1) | (0,0,2) |
 ----------------------------- batch 0
 (0,1,0) | (0,1,1) | (0,1,2) |
 -------------------------------------
 (1,0,0) | (1,0,1) | (1,0,2) |
 ----------------------------- batch 1
 (1,1,0) | (1,1,1) | (1,1,2) |
 -----------------------------
 , where the batch size is 2, N is 3 and K is 2
 The stride (batch_stride_C) between the first element of two batches is k
 
 
 */
 
 cudaError_t cutlass_strided_batched_sgemm(
   int m, 
   int n,
   int k,
   float alpha,
   float const *A,
   int lda,
   long long int batch_stride_A,
   float const *B,
   int ldb,
   long long int batch_stride_B,
   float *C,
   int ldc,
   long long int batch_stride_C,
   float beta,
   int batch_count) {
 
   using Gemm = cutlass::gemm::device::GemmBatched<
     float, cutlass::layout::RowMajor,
     float, cutlass::layout::RowMajor,
     float, cutlass::layout::RowMajor
   >;
 
   Gemm gemm_op;
 
   cutlass::Status status = gemm_op({
     {m, n, k},
     {A, lda}, 
     batch_stride_A,
     {B, ldb}, 
     batch_stride_B,
     {C, ldc}, 
     batch_stride_C,
     {C, ldc}, 
     batch_stride_C,
     {alpha, beta},
     batch_count
   });
 
   if (status != cutlass::Status::kSuccess) {
     return cudaErrorUnknown;
   }
 
   return cudaSuccess;
 }
 
 template<typename T> 
 cudaError_t strided_batched_gemm_nn_reference(
   int m,
   int n,
   int k,
   T alpha,
   std::vector<T> const &A, 
   int lda,
   long long int batch_stride_A,
   std::vector<T> const &B, 
   int ldb,
   long long int batch_stride_B,
   std::vector<T> &C, 
   int ldc,
   long long int batch_stride_C,
   T beta,
   int batch_count) {
   /*
   strided batched gemm NN
   */
 
   for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
     for (int n_idx = 0; n_idx < n; n_idx++) {
       for (int m_idx = 0; m_idx < m; m_idx++) {
         T accum = beta * C[batch_idx * batch_stride_C + m_idx * ldc + n_idx];
         for (int k_idx = 0; k_idx < k; k_idx++) {
           accum += alpha 
             * A[batch_idx * batch_stride_A + m_idx * lda + k_idx]
             * B[batch_idx * batch_stride_B + k_idx * ldb + n_idx];
         }
         C[batch_idx * batch_stride_C + m_idx * ldc + n_idx] = accum;
       }
     }
   }
 
   return cudaSuccess;
 }
 
 int main() {
 
   // Arbitrary problem size
   int const m = 520;
   int const n = 219;
   int const k = 129;
   int const batch_count = 17;
 
   // A, B are non-transpose, column major
   int const lda = k;
   int const ldb = n;
   int const ldc = n;

   int const count_A = batch_count * lda * m;
   int const count_B = batch_count * ldb * k;
   int const count_C = batch_count * ldc * m;
 
   // the memory is batched along K dimension
   long long int batch_stride_A = static_cast<long long int>(m) * static_cast<long long int>(k);
   long long int batch_stride_B = static_cast<long long int>(k) * static_cast<long long int>(n);
   long long int batch_stride_C = static_cast<long long int>(m) * static_cast<long long int>(n);
 
   // alpha and beta
   float alpha = 1.0f;
   float beta = 2.0f;
 
   cudaError_t result = cudaSuccess;
 
   // allocate the host memory
   std::vector<float> host_A(count_A);
   std::vector<float> host_B(count_B);
   std::vector<float> host_C(count_C);
   std::vector<float> result_C(count_C);
 
   // allocate the device memory
   float *A;
   float *B;
   float *C;
 
   CUDACHECK(cudaMalloc(&A, count_A * sizeof(float)));
   CUDACHECK(cudaMalloc(&B, count_B * sizeof(float)));
   CUDACHECK(cudaMalloc(&C, count_C * sizeof(float)));
 
   // Limit range to avoid floating-point errors
   int const kRange = 8;
 
   // fill A
   for (int b_idx = 0; b_idx < batch_count; b_idx++) {
     for (int col_idx = 0; col_idx < k; col_idx++) {
       for (int row_idx = 0; row_idx < m; row_idx++) {
         host_A[row_idx + col_idx * lda + b_idx * lda * k] = static_cast<float>((row_idx + col_idx * lda + b_idx * lda * k) % kRange);
       }
     }
   }
   // fill B
   for (int b_idx = 0; b_idx < batch_count; b_idx++) {
     for (int col_idx = 0; col_idx < n; col_idx++) {
       for (int row_idx = 0; row_idx < k; row_idx++) {
         host_B[row_idx + col_idx * ldb + b_idx * k] = static_cast<float>(((n + k * ldb + batch_count * k) - (row_idx + col_idx * ldb + b_idx * k)) % kRange);
       }
     }
   }
   // fill C
   for (int b_idx = 0; b_idx < batch_count; b_idx++) {
     for (int col_idx = 0; col_idx < n; col_idx++) {
       for (int row_idx = 0; row_idx < m; row_idx++) {
         host_C[row_idx + col_idx * ldc + b_idx * ldc * n] = 1.f;
       }
     }
   }
 
   // ref memory
   std::vector<float> ref_A(host_A);
   std::vector<float> ref_B(host_B);
   std::vector<float> ref_C(host_C);
   // copy host memory to device
   CUDACHECK(cudaMemcpy(A, host_A.data(), count_A * sizeof(float), cudaMemcpyHostToDevice));
   CUDACHECK(cudaMemcpy(B, host_B.data(), count_B * sizeof(float), cudaMemcpyHostToDevice));
   CUDACHECK(cudaMemcpy(C, host_C.data(), count_C * sizeof(float), cudaMemcpyHostToDevice));

   // run cutlass
   result = cutlass_strided_batched_sgemm(
     m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C,
     beta, batch_count);
   if (result != cudaSuccess)
     return result;
 
   // copy device memory to host
   CUDACHECK(cudaMemcpy(result_C.data(), C, count_C * sizeof(float), cudaMemcpyDeviceToHost));
 
   //compare with reference code
   result = strided_batched_gemm_nn_reference(m, n, k, alpha, ref_A, lda, batch_stride_A, ref_B, ldb, batch_stride_B, ref_C, ldc, batch_stride_C,
     beta, batch_count);
   if (result != cudaSuccess)
     return result;
 
   // Expect bit-level accuracy for this simple example
   if (ref_C != result_C) {
     std::cout << "CUTLASS strided batched gemm does not run correctly" << std::endl;
     return cudaErrorUnknown;
   }
 
   // free memory
   CUDACHECK(cudaFree(A));
   CUDACHECK(cudaFree(B));
   CUDACHECK(cudaFree(C));
 
   if (result == cudaSuccess) {
     std::cout << "Passed." << std::endl;
   }
 
   // Exit.
   return result == cudaSuccess ? 0 : -1;
 }
 