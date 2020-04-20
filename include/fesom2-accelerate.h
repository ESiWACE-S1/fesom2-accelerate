
#include <cstdlib>
#include <iostream>
#include <limits>
#include <driver_types.h>
#include <cuda_runtime_api.h>

using real_type = double;

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


/**
 CPU reference implementation of step a1 of FCT_ALE.
 This step computes the maximum and minimum between the old solution and the updated low-order solution per node.

 @param nodes The number of nodes
 @param nLevels_nod2D Array containing the number of horizontal levels per node
 @param fct_ttf_max Computed maximum
 @param fct_ttf_min Computed minimum
 @param fct_low_order New low order solution of fct
 @param ttf Old solution
*/
void fct_ale_a1_reference(int nodes, int * nLevels_nod2D, real_type * fct_ttf_max, real_type * fct_ttf_min,  real_type * fct_low_order, real_type * ttf);