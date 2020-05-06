
#include <fesom2-accelerate.h>
#include <vector_types.h>


extern __global__ void fct_ale_a1(const double * __restrict__ fct_low_order, const double * __restrict__ ttf, const int * __restrict__ nLevels, double * fct_ttf_max, double * fct_ttf_min);

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

void fct_ale_a1_accelerated(int nNodes, struct gpuMemory * nLevels_nod2D, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min,  struct gpuMemory * fct_low_order, struct gpuMemory * ttf, bool synchronous, cudaStream_t stream)
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
