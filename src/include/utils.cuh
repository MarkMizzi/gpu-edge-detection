#pragma once

#include <cstdlib>
#include <iostream>
#include <limits>

/** This is a wrapper around std::numeric_limits that can be used inside a CUDA kernel.
 */
template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
struct numeric_limits
{
    static constexpr T min = std::numeric_limits<T>::min();
    static constexpr T max = std::numeric_limits<T>::max();
};

#define checkCudaErrors(call)                                                                                             \
    {                                                                                                                     \
        cudaError_t err = call;                                                                                           \
        if (err != cudaSuccess)                                                                                           \
        {                                                                                                                 \
            std::cout << "CUDA error at " << __FILE__ << " " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                                                           \
        }                                                                                                                 \
    }

__device__ __host__ int floor_log_2(size_t x)
{
#ifdef __CUDA_ARCH__
    return sizeof(size_t) * 8 - __clzll(x);
#else
    return sizeof(size_t) * 8 - __builtin_clzll(x);
#endif
}
