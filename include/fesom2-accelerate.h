
#include <cstdlib>
#include <driver_types.h>
#include <iostream>
#include <cuda_runtime_api.h>

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
 @param stream The CUDA stream associated with the transfer
 @param synchronous A boolean value to control synchronization

 @return The boolean value true if there are no errors, false otherwise
*/
inline bool transferToDevice(gpuMemory & buffer, cudaStream_t stream = 0, bool synchronous = true)
{
    cudaError_t status = cudaSuccess;
    status = cudaMemcpy(buffer.device_pointer, buffer.host_pointer, buffer.size, cudaMemcpyHostToDevice);
    if ( synchronous )
    {
        cudaStreamSynchronize(stream);
    }
    return errorHandling(status);
}

/**
 Function to transfer data from the device to the host.

 @param buffer A reference to the memory
 @param stream The CUDA stream associated with the transfer
 @param synchronous A boolean value to control synchronization

 @return The boolean value true if there are no errors, false otherwise
*/
inline bool transferToHost(gpuMemory & buffer, cudaStream_t stream = 0, bool synchronous = true)
{
    cudaError_t status= cudaSuccess;
    status = cudaMemcpy(buffer.host_pointer, buffer.device_pointer, buffer.size, cudaMemcpyDeviceToHost);
    if ( synchronous )
    {
        cudaStreamSynchronize(stream);
    }
    return errorHandling(status);
}
