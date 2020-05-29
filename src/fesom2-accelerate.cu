#include <fesom2-accelerate.h>
#include <vector_types.h>
#include <iostream>

// CUDA kernels
extern __global__ void fct_ale_a1(const int maxLevels, const double * __restrict__ fct_low_order, const double * __restrict__ ttf, const int * __restrict__ nLevels, double * fct_ttf_max, double * fct_ttf_min);
extern __global__ void fct_ale_a2(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ elementNodes, double2 * __restrict__ UV_rhs, const double * __restrict__ fct_ttf_max, const double * __restrict__ fct_ttf_min);
extern __global__ void fct_ale_a2b(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ elementNodes, double * __restrict__ UV_rhs, const double * __restrict__ fct_ttf_max, const double * __restrict__ fct_ttf_min, double bignumber);
extern __global__ void fct_ale_a3(const int maxLevels, const int maxElements, const int * __restrict__ nLevels, const int * __restrict__ elements_in_node, const int * __restrict__ number_elements_in_node, const double2 * __restrict__ UV_rhs, double * __restrict__ fct_ttf_max, double * __restrict__ fct_ttf_min, const double * __restrict__ fct_lo);
extern __global__ void fct_ale_b1_vertical(const int maxLevels, const int * __restrict__ nLevels, const double * __restrict__ fct_adf_v, double * __restrict__ fct_plus, double * __restrict__ fct_minus);
extern __global__ void fct_ale_pre_comm(   const int max_levels, const int num_nodes, const int max_num_elems, const int * __restrict__ node_levels, const int * __restrict__ elem_levels, const int * __restrict__ node_elems, const int * __restrict__ node_num_elems, const int * __restrict__ elem_nodes, const double * __restrict__ fct_low_order, const double * __restrict__ ttf, const double * __restrict__ fct_adf_v, const double * __restrict__ fct_adf_h, double * __restrict__ UVrhs, double * __restrict__ fct_ttf_max, double * __restrict__ fct_ttf_min, double * __restrict__ tvert_max, double * __restrict__ tvert_min, double * __restrict__ fct_plus, double * __restrict__ fct_minus, const double bignr);

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

void fct_ale_a1_accelerated(const int maxLevels, const int nNodes, struct gpuMemory * nLevels_nod2D, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min,  struct gpuMemory * fct_low_order, struct gpuMemory * ttf, bool synchronous, cudaStream_t stream)
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
    fct_ale_a1<<< dim3(nNodes), dim3(32) >>>(maxLevels, reinterpret_cast<real_type *>(fct_low_order->device_pointer), reinterpret_cast<real_type *>(ttf->device_pointer), reinterpret_cast<int *>(nLevels_nod2D->device_pointer), reinterpret_cast<real_type *>(fct_ttf_max->device_pointer), reinterpret_cast<real_type *>(fct_ttf_min->device_pointer));
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

void fct_ale_a2_accelerated(const int maxLevels, const int nElements, const struct gpuMemory * nLevels_elem, struct gpuMemory * elementNodes, struct gpuMemory * UV_rhs, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min, bool synchronous, cudaStream_t stream)
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
    fct_ale_a2<<< dim3(nElements), dim3(32) >>>(maxLevels, reinterpret_cast<int *>(nLevels_elem->device_pointer), reinterpret_cast<int *>(elementNodes->device_pointer), reinterpret_cast<real2_type *>(UV_rhs->device_pointer), reinterpret_cast<real_type *>(fct_ttf_max->device_pointer), reinterpret_cast<real_type *>(fct_ttf_min->device_pointer));
    status = transferToHost(*UV_rhs, synchronous, stream);
    if ( !status )
    {
        return;
    }
}

void fct_ale_a1_a2_accelerated(const int maxLevels, const int nNodes, const int nElements, struct gpuMemory * nLevels_nod2D, struct gpuMemory * nLevels_elem, struct gpuMemory * elementNodes, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min, struct gpuMemory * fct_low_order, struct gpuMemory * ttf, struct gpuMemory * UV_rhs, bool synchronous, cudaStream_t stream)
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
    fct_ale_a1<<< dim3(nNodes), dim3(32) >>>(maxLevels, reinterpret_cast<real_type *>(fct_low_order->device_pointer), reinterpret_cast<real_type *>(ttf->device_pointer), reinterpret_cast<int *>(nLevels_nod2D->device_pointer), reinterpret_cast<real_type *>(fct_ttf_max->device_pointer), reinterpret_cast<real_type *>(fct_ttf_min->device_pointer));
    fct_ale_a2<<< dim3(nElements), dim3(32) >>>(maxLevels, reinterpret_cast<int *>(nLevels_elem->device_pointer), reinterpret_cast<int *>(elementNodes->device_pointer), reinterpret_cast<real2_type *>(UV_rhs->device_pointer), reinterpret_cast<real_type *>(fct_ttf_max->device_pointer), reinterpret_cast<real_type *>(fct_ttf_min->device_pointer));
    status = transferToHost(*UV_rhs, synchronous, stream);
    if ( !status )
    {
        return;
    }
}

void transfer_mesh_(void** ret, int* host_ptr, int* size, int* istat)
{
    struct gpuMemory* gpumem = allocate((void*)host_ptr, (*size) * sizeof(int));
    if ( transferToDevice(*gpumem) )
    {
        *ret = (void*)gpumem;
        *istat = 0;
    }
    else
    {
        *ret = nullptr;
        *istat = 1;
    }
}

void alloc_var_(void** ret, real_type* host_ptr, int* size, int* istat)
{
    struct gpuMemory* gpumem = allocate((void*)host_ptr, (*size) * sizeof(real_type));
    *istat = (gpumem == nullptr)?1:0;
    *ret = (void*)gpumem;
}

void reserve_var_(void** ret, int* size, int* istat)
{
    struct gpuMemory* gpumem = allocate(nullptr, (*size) * sizeof(real_type));
    *istat = (gpumem == nullptr)?1:0;
    *ret = (void*)gpumem;
}

std::ostream& operator << (std::ostream& os, const gpuMemory& gpumem)
{
    os<<"host at:"<<gpumem.host_pointer<<", dev at:"<<gpumem.device_pointer<<", size: "<<gpumem.size;
    return os;
}

void fct_ale_pre_comm_acc_( int* alg_state, void** fct_ttf_max, void**  fct_ttf_min, 
                            real_type*  fct_plus, real_type*  fct_minus, real_type* tvert_max, 
                            real_type*  tvert_min, void** ttf, real_type* ttf_vals, void** fct_LO, real_type*  fct_adf_v,
                            real_type* fct_adf_h, void** UV_rhs, real_type* area_inv, int* myDim_nod2D, 
                            int* eDim_nod2D, int* myDim_elem2D, int* myDim_edge2D, int* nl, void** nlevels_nod2D, 
                            void** nlevels_elem2D, void** elem2D_nodes, int* nod_in_elem2D_num, int* nod_in_elem2D, 
                            int* nod_in_elem2D_dim, int* nod2D_edges, int* elem2D_edges, int* vlimit, 
                            real_type* flux_eps, real_type* bignumber, real_type* dt)
{
    *alg_state = 0;
    bool status = true;
    int nNodes = (*myDim_nod2D) + (*eDim_nod2D);

    status = transferToDevice(*static_cast<gpuMemory*>(*fct_LO));
    if ( !status )
    {
        return;
    }

    struct gpuMemory* ttf_gpu = static_cast<gpuMemory*>(*ttf);
    ttf_gpu->host_pointer = (void*)ttf_vals;
    status = transferToDevice(*ttf_gpu);
    if ( !status )
    {
        return;
    }

    real_type* fct_lo_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_LO)->device_pointer);
    real_type* ttf_dev    = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*ttf)->device_pointer);
    real_type* fct_ttf_max_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_ttf_max)->device_pointer);
    real_type* fct_ttf_min_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_ttf_min)->device_pointer);
    real_type* UV_rhs_dev    = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*UV_rhs)->device_pointer);
    int* nlevels_nod2D_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*nlevels_nod2D)->device_pointer);
    int* nlevels_elem2D_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*nlevels_elem2D)->device_pointer);
    int* elem2D_nodes_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*elem2D_nodes)->device_pointer);

    int maxLevels = *nl - 1;
    fct_ale_a1<<< dim3(nNodes), dim3(32) >>>(maxLevels, fct_lo_dev, ttf_dev, nlevels_nod2D_dev, fct_ttf_max_dev, 
                                             fct_ttf_min_dev);
    *alg_state = 1;
    fct_ale_a2b<<< dim3(*myDim_elem2D), dim3(32) >>>(maxLevels,nlevels_elem2D_dev, elem2D_nodes_dev, UV_rhs_dev, 
                                                    fct_ttf_max_dev, fct_ttf_min_dev, *bignumber);
    status = transferToHost(*static_cast<gpuMemory*>(*UV_rhs));
    if ( !status )
    {
        return;
    }
    *alg_state = 2;
}

void fct_ale_pre_comm_acc2_( int* alg_state, void** fct_ttf_max, void**  fct_ttf_min, 
    void**  fct_plus, void**  fct_minus, void** tvert_max, 
    void**  tvert_min, void** ttf, real_type* ttf_vals, void** fct_LO, void**  fct_adf_v,
    void** fct_adf_h, void** UV_rhs, real_type* area_inv, int* myDim_nod2D, 
    int* eDim_nod2D, int* myDim_elem2D, int* myDim_edge2D, int* nl, void** nlevels_nod2D, 
    void** nlevels_elem2D, void** elem2D_nodes, void** nod_in_elem2D_num, void** nod_in_elem2D, 
    int* nod_in_elem2D_dim, int* nod2D_edges, int* elem2D_edges, int* vlimit, 
    real_type* flux_eps, real_type* bignumber, real_type* dt)
{
    *alg_state = 0;
    bool status = true;
    int nNodes = (*myDim_nod2D) + (*eDim_nod2D);

    status = transferToDevice(*static_cast<gpuMemory*>(*fct_LO)) and 
             transferToDevice(*static_cast<gpuMemory*>(*fct_adf_v));
    if ( !status )
    {
        return;
    }

    struct gpuMemory* ttf_gpu = static_cast<gpuMemory*>(*ttf);
    ttf_gpu->host_pointer = (void*)ttf_vals;
    status = transferToDevice(*ttf_gpu);
    if ( !status )
    {
        std::cerr<<"Error in transfer of fct_ttf to GPU"<<std::endl;
        return;
    }

    int maxLevels = *nl - 1;
    int maxnElems = *nod_in_elem2D_dim;
    int* nlevels_nod2D_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*nlevels_nod2D)->device_pointer);
    int* nlevels_elem2D_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*nlevels_elem2D)->device_pointer);
    int* node_elems_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*nod_in_elem2D)->device_pointer);
    int* node_num_elems_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*nod_in_elem2D_num)->device_pointer);
    int* elem2D_nodes_dev = reinterpret_cast<int*>(static_cast<gpuMemory*>(*elem2D_nodes)->device_pointer);
    real_type* fct_lo_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_LO)->device_pointer);
    real_type* ttf_dev    = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*ttf)->device_pointer);
    real_type* fct_adf_v_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_adf_v)->device_pointer);
    real_type* UV_rhs_dev    = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*UV_rhs)->device_pointer);
    real_type* fct_ttf_max_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_ttf_max)->device_pointer);
    real_type* fct_ttf_min_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_ttf_min)->device_pointer);
    real_type* tvert_max_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*tvert_max)->device_pointer);
    real_type* tvert_min_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*tvert_min)->device_pointer);
    real_type* fct_plus_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_plus)->device_pointer);
    real_type* fct_min_dev = reinterpret_cast<real_type*>(static_cast<gpuMemory*>(*fct_minus)->device_pointer);

    fct_ale_pre_comm<<< dim3(nNodes), dim3(32) >>>( maxLevels,
                                                    *myDim_nod2D,
                                                    maxnElems,
                                                    nlevels_nod2D_dev,
                                                    nlevels_elem2D_dev,
                                                    node_elems_dev,
                                                    node_num_elems_dev,
                                                    elem2D_nodes_dev,
                                                    fct_lo_dev, 
                                                    ttf_dev,
                                                    fct_adf_v_dev,
                                                    nullptr, //Not used yet
                                                    UV_rhs_dev,
                                                    fct_ttf_max_dev, 
                                                    fct_ttf_min_dev,
                                                    tvert_max_dev,
                                                    tvert_min_dev,
                                                    fct_plus_dev,
                                                    fct_min_dev,
						    *bignumber);
    status =    transferToHost(*static_cast<gpuMemory*>(*UV_rhs)) and 
                transferToHost(*static_cast<gpuMemory*>(*fct_plus)) and 
                transferToHost(*static_cast<gpuMemory*>(*fct_minus));
    if ( !status )
    {
        std::cerr<<"Error in transfer of UV_rhs or fct_plus or fct_minus to CPU"<<std::endl;
        *alg_state = 0;
        return;
    }
    *alg_state = 4;
}
