
#include <fesom2-accelerate.h>
#include <vector_types.h>


// CUDA kernels
extern __global__ void fct_ale_a1(const double * __restrict__ fct_low_order, const double * __restrict__ ttf, const int * __restrict__ nLevels, double * fct_ttf_max, double * fct_ttf_min);
extern __global__ void fct_ale_a2(const int * __restrict__ nLevels, const int * __restrict__ elementNodes, double2 * __restrict__ UV_rhs, const double * __restrict__ fct_ttf_max, const double * __restrict__ fct_ttf_min);

struct gpuMemory * allocate(void * hostMemory, std::size_t size)
{
    cudaError_t status = cudaSuccess;
    struct gpuMemory * allocatedMemory = new struct gpuMemory;

    allocatedMemory->host_pointer = hostMemory;
    allocatedMemory->size = size;
    status = cudaMalloc(&(allocatedMemory->device_pointer), size);
    if ( !errorHandling(status) )
    {
        delete allocatedMemory;
        return nullptr;
    }
    return allocatedMemory;
}

void fct_ale_a1_accelerated(const int nNodes, struct gpuMemory * nLevels_nod2D, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min,  struct gpuMemory * fct_low_order, struct gpuMemory * ttf, bool synchronous, cudaStream_t stream)
{
    bool status = true;

    status = transferToDevice(*fct_low_order, synchronous, stream);
    if ( !status )
    {
        return;
    }
    status = transferToDevice(*ttf, synchronous, stream);
    if ( !status )
    {
        return;
    }
    fct_ale_a1<<< dim3(nNodes), dim3(32) >>>(reinterpret_cast<real_type *>(fct_low_order->device_pointer), reinterpret_cast<real_type *>(ttf->device_pointer), reinterpret_cast<int *>(nLevels_nod2D->device_pointer), reinterpret_cast<real_type *>(fct_ttf_max->device_pointer), reinterpret_cast<real_type *>(fct_ttf_min->device_pointer));
    status = transferToHost(*fct_ttf_max, synchronous, stream);
    if ( !status )
    {
        return;
    }
    status = transferToHost(*fct_ttf_min, synchronous, stream);
    if ( !status )
    {
        return;
    }
}

void fct_ale_a2_accelerated(const int nElements, const struct gpuMemory * nLevels_elem, struct gpuMemory * elementNodes, struct gpuMemory * UV_rhs, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min, bool synchronous, cudaStream_t stream)
{
    bool status = true;

    status = transferToDevice(*fct_ttf_max, synchronous, stream);
    if ( !status )
    {
        return;
    }
    status = transferToDevice(*fct_ttf_min, synchronous, stream);
    if ( !status )
    {
        return;
    }
    fct_ale_a2<<< dim3(nElements), dim3(32) >>>(reinterpret_cast<int *>(nLevels_elem->device_pointer), reinterpret_cast<int *>(elementNodes->device_pointer), reinterpret_cast<real2_type *>(UV_rhs->device_pointer), reinterpret_cast<real_type *>(fct_ttf_max->device_pointer), reinterpret_cast<real_type *>(fct_ttf_min->device_pointer));
    status = transferToHost(*UV_rhs, synchronous, stream);
    if ( !status )
    {
        return;
    }
}
