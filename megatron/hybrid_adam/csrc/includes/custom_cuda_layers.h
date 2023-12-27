#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "context.h"

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define MAX_THREADS 1024
#define THREADS 256

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 32

// Maximum sequence-length support based on the number of threads (2048) allowed in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define MAX_WARP_NUM 32

#define MAX_REGISTERS 256

#define MAX_REG 256

#define WARP_SIZE_BITS 5

void launch_param_update(const float* input, __half* output, int size, cudaStream_t stream);
void launch_param_update_half(const float* input, __half* output, int size, cudaStream_t stream);
