#ifndef ATOMIC_ADD_FIX_H_
#define ATOMIC_ADD_FIX_H_
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double*, double);
#endif
#endif
