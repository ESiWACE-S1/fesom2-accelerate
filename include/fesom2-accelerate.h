
#include <cstdlib>
#include <iostream>
#include <limits>
#ifdef __CUDACC__
#include <driver_types.h>
#include <cuda_runtime_api.h>
#endif
using real_type = double;
using real2_type = double2;

#ifdef __CUDACC__
/**
 A structure to map GPU and host memory.
*/
struct gpuMemory {
    /** Pointer to host memory. */
    void * host_pointer;
    /** Pointer to device memory. */
    void * device_pointer;
    /** Size of the allocated memory, in bytes. */
    std::size_t size;
};

/**
 Simple function to handle CUDA errors.
 If an error is detected, a debug message is printed in the standard error.

 @param error The error code, returned by the CUDA runtime

 @return The boolean value true if there are no errors, false otherwise
*/
inline bool errorHandling(cudaError_t error)
{
    if ( error != cudaSuccess )
    {
        std::cerr << "CUDA error \"" << cudaGetErrorName(error) << "\": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

/**
 Allocate memory on the GPU.
 
 @param hostMemory Pointer to host memory
 @param size The size, in bytes, of allocated host memory

 @return A pointer to struct gpuMemory containing the allocated device pointer
*/
struct gpuMemory * allocate(void * hostMemory, std::size_t size);

/**
 Function to transfer data from the host to the device.

 @param buffer A reference to the memory
 @param synchronous A boolean value to control synchronization
 @param stream The CUDA stream associated with the transfer

 @return The boolean value true if there are no errors, false otherwise
*/
inline bool transferToDevice(gpuMemory & buffer, bool synchronous = true, cudaStream_t stream = (cudaStream_t) 0)
{
    cudaError_t status = cudaSuccess;
    if ( synchronous )
    {
        status = cudaMemcpy(buffer.device_pointer, buffer.host_pointer, buffer.size, cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMemcpyAsync(buffer.device_pointer, buffer.host_pointer, buffer.size, cudaMemcpyHostToDevice, stream);
    }
    return errorHandling(status);
}

/**
 Function to transfer data from the device to the host.

 @param buffer A reference to the memory
 @param synchronous A boolean value to control synchronization
 @param stream The CUDA stream associated with the transfer

 @return The boolean value true if there are no errors, false otherwise
*/
inline bool transferToHost(gpuMemory & buffer, bool synchronous = true, cudaStream_t stream = (cudaStream_t) 0)
{
    cudaError_t status= cudaSuccess;
    if ( synchronous )
    {
        status = cudaMemcpy(buffer.host_pointer, buffer.device_pointer, buffer.size, cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpyAsync(buffer.host_pointer, buffer.device_pointer, buffer.size, cudaMemcpyDeviceToHost, stream);
    }
    
    return errorHandling(status);
}
#endif

extern "C"
{
/**
 CPU reference implementation of step a1 of FCT_ALE.
 This step computes the maximum and minimum between the old solution and the updated low-order solution per node.

 @param nNodes The number of nodes
 @param nLevels_nod2D Array containing the number of vertical levels per node
 @param maxLevels_ptr Maximum number of levels per node
 @param fct_ttf_max Computed maximum
 @param fct_ttf_min Computed minimum
 @param fct_low_order New low order solution of fct
 @param ttf Old solution
*/
void fct_ale_a1_reference_( int * nNodes, int * nLevels_nod2D, int * maxLevels_ptr, real_type * fct_ttf_max, 
                            real_type * fct_ttf_min,  real_type * fct_low_order, real_type * ttf);

#ifdef __CUDACC__
/**
 GPU CUDA implementation of step a1 of FCT_ALE.
 This step computes the maximum and minimum between the old solution and the updated low-order solution per node.

 @param nNodes The number of nodes
 @param nLevels_nod2D Array containing the number of vertical levels per node
 @param fct_ttf_max Computed maximum
 @param fct_ttf_min Computed minimum
 @param fct_low_order New low order solution of fct
 @param ttf Old solution
 @param synchronous A boolean value to control synchronization
 @param stream The CUDA stream associated with the transfer
*/
void fct_ale_a1_accelerated(const int nNodes, struct gpuMemory * nLevels_nod2D, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min,  struct gpuMemory * fct_low_order, struct gpuMemory * ttf, bool synchronous = true, cudaStream_t stream = (cudaStream_t) 0);

/**
 GPU CUDA implementation of step a2 of FCT_ALE.
 Computes the admissible increments on elements.

 @param nElements The number of elements
 @param nLevels_elem Array containing the number of vertical levels per element
 @param elementNodes Array containing the ID of the three nodes of each element
 @param UV_rhs Computed admissible increments
 @param fct_ttf_max Maximum computed in step a1
 @param fct_ttf_min Minimum computed in step a1
 @param synchronous A boolean value to control synchronization
 @param stream The CUDA stream associated with the transfer
*/
void fct_ale_a2_accelerated(const int nElements, struct gpuMemory * nLevels_elem, struct gpuMemory * elementNodes, struct gpuMemory * UV_rhs, struct gpuMemory * fct_ttf_max, struct gpuMemory * fct_ttf_min, bool synchronous = true, cudaStream_t stream = (cudaStream_t) 0);
#endif

/**
 CPU reference implementation of step a2 of FCT_ALE.
 Computing maximum and minimum bounds per element.

 @param nElement Number of elements
 @param maxLevels_ptr Maximum number of levels per node
 @param nLevels Array containing the number of vertical levels per element
 @param UV_rhs Three dimensional array containing bound for each element
 @param elem2D_nodes Array containing the three nodes of an element
 @param fct_ttf_max Previously computed maximum
 @param fct_ttf_min Previously computed minimum
*/
void fct_ale_a2_reference_( int * nElements, int * maxLevels, int * nLevels, real_type * UV_rhs, 
                            int * elem2D_nodes, real_type * fct_ttf_max, real_type * fct_ttf_min,
                            real_type * bignumber );


void fct_ale_a3_reference_( int * nNodes2D, int * nLevels_nod2D, int * nl, real_type * fct_ttf_max, 
                            real_type * fct_ttf_min,  real_type * fct_LO, 
                            real_type * tvert_max, real_type * tvert_min, real_type * UV_rhs, 
                            real_type * fct_plus, real_type * fct_minus, real_type * fct_adf_v,
                            int * nod_in_elem2D, int * nod_in_elem2D_num, int * nod_in_elem2D_dim );


void fct_ale_pre_comm_( int* alg_state, real_type* fct_ttf_max, real_type*  fct_ttf_min, 
                        real_type*  fct_plus, real_type*  fct_minus, real_type* tvert_max, 
                        real_type*  tvert_min, real_type* ttf, real_type* fct_LO, real_type*  fct_adf_v, 
                        real_type* UV_rhs, int* myDim_nod2D, int* eDim_nod2D, int* myDim_elem2D, 
                        int* myDim_edge2D, int* nl, int* nlevels_nod2D, int* nlevels, int* elem2D_nodes, 
                        int* nod_in_elem2D_num, int* nod_in_elem2D, int* nod_in_elem2D_dim, int* vlimit, real_type* flux_eps,
                        real_type * bignumber);
}