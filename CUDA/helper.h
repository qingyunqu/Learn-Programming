#pragma once

#include <vector>
#include <cstdlib>
#include <algorithm>

template <typename T>
void TensorFillRandom(T* input, size_t total_size, int seed = 1, T max = 1, T min = 0) {
    srand(seed);
    for(size_t i = 0; i < total_size; i++) {
        input[i] = (T)((rand() / (float)2147483647) * (max - min) + min);
    }
}


template <typename T>
bool TensorEquals(T* input, T* input_ref, size_t total_size, T eps = 0.0001) {
    for(size_t i = 0; i < total_size; i++) {
        if (std::abs(input[i] - input_ref[i]) > eps) {
            printf("output: %f\tref:%f\n", input[i], input_ref[i]);
            return false;
        }
    }
    return true;
}