#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TOTAL_THREADS_DENSE 512
#define TOTAL_THREADS_SPARSE 1024

inline int opt_n_threads(int work_size)
{
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS_DENSE), 1);
}

inline dim3 opt_block_config(int x, int y)
{
    const int x_threads = opt_n_threads(x);
    const int y_threads = max(min(opt_n_threads(y), TOTAL_THREADS_DENSE / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}

// from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#define CUDA_CHECK_ERRORS()                                                                        \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (cudaSuccess != err)                                                                    \
        {                                                                                          \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",                         \
                    cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, __FILE__);             \
            exit(-1);                                                                              \
        }                                                                                          \
    } while (0)

#endif
