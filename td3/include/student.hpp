/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.hpp
* Author: Maxime MARIA
*/

#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"
#include "chronoGPU.hpp"
#include "chronoCPU.hpp"

namespace IMAC
{
	const uint MAX_NB_THREADS = 1024; // En dur, changer si GPU plus ancien ;-)
    const uint DEFAULT_NB_BLOCKS = 32768;

    enum
    {
        KERNEL_EX1 = 0,
        KERNEL_EX2,
        KERNEL_EX3,
        KERNEL_EX4,
        KERNEL_EX5
    };
	
	// ==================================================== EX 1
	__global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax);
    
	// return a uint2 with x: dimBlock / y: dimGrid
    template<uint kernelType>
    uint2 configureKernel(const uint sizeArray)
    {
        uint2 dimBlockGrid; // x: dimBlock / y: dimGrid

		// Configure number of threads/blocks
		switch(kernelType)
		{
			case KERNEL_EX1:
				dimBlockGrid.x = MAX_NB_THREADS; 
				dimBlockGrid.y = DEFAULT_NB_BLOCKS;
			break;
			case KERNEL_EX2:
				/// TODO EX 2
			break;
			case KERNEL_EX3:
				/// TODO EX 3
			break;
			case KERNEL_EX4:
				/// TODO EX 4
			break;
			case KERNEL_EX5:
				/// TODO EX 5
			break;
            default:
                throw std::runtime_error("Error configureKernel: unknown kernel type");
		}
		verifyDimGridBlock( dimBlockGrid.y, dimBlockGrid.x, sizeArray ); // Are you reasonable ?
        
        return dimBlockGrid;
    }

    // Launch kernel number 'kernelType' and return float2 for timing (x:device,y:host)    
    template<uint kernelType>
    float2 reduce(const uint nbIterations, const uint *const dev_array, const uint size, uint &result)
	{
        const uint2 dimBlockGrid = configureKernel<kernelType>(size); // x: dimBlock / y: dimGrid

		// Allocate arrays (host and device) for partial result
		/// TODO
		std::vector<uint> host_partialMax(0); // REPLACE SIZE !
		const size_t bytesPartialMax = 0; // REPLACE BYTES !
		const size_t bytesSharedMem = 0; // REPLACE BYTES !
		
		uint *dev_partialMax;
		HANDLE_ERROR(cudaMalloc((void**) &dev_partialMax, bytesPartialMax ) );

		std::cout 	<< "Computing on " << dimBlockGrid.y << " block(s) and " 
					<< dimBlockGrid.x << " thread(s) "
					<<"- shared mem size = " << bytesSharedMem << std::endl;

		ChronoGPU chrGPU;
		float2 timing = { 0.f, 0.f }; // x: timing GPU, y: timing CPU
		// Average timing on 'loop' iterations
		for (uint i = 0; i < nbIterations; ++i)
		{
			chrGPU.start();
			switch(kernelType) // Evaluated at compilation time
			{
				case KERNEL_EX1:
					/// TODO EX 2
					/// maxReduce_ex1<<<>>>();
					std::cout << "Not implemented !" << std::endl;
				break;
				case KERNEL_EX2:
					/// TODO EX 2
					std::cout << "Not implemented !" << std::endl;
				break;
				case KERNEL_EX3:
					/// TODO EX 3
					std::cout << "Not implemented !" << std::endl;
				break;
				case KERNEL_EX4:
					/// TODO EX 4
					std::cout << "Not implemented !" << std::endl;
				break;
				case KERNEL_EX5:
					/// TODO EX 5
					std::cout << "Not implemented !" << std::endl;
				break;
                default:
		            cudaFree(dev_partialMax);
                    throw("Error reduce: unknown kernel type.");
			}
			chrGPU.stop();
			timing.x += chrGPU.elapsedTime();
		}
		timing.x /= (float)nbIterations; // Stores time for device

		// Retrieve partial result from device to host
		HANDLE_ERROR(cudaMemcpy(host_partialMax.data(), dev_partialMax, bytesPartialMax, cudaMemcpyDeviceToHost));

		cudaFree(dev_partialMax);

        // Check for error
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			throw std::runtime_error(cudaGetErrorString(err));
		}
		
		ChronoCPU chrCPU;
		chrCPU.start();

		// Finish on host
		for (int i = 0; i < host_partialMax.size(); ++i)
		{
			result = std::max<uint>(result, host_partialMax[i]);
		}
		
		chrCPU.stop();

		timing.y = chrCPU.elapsedTime(); // Stores time for host
		
        return timing;
	}  
    
    void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations);

    void printTiming(const float2 timing);
    void compare(const uint resGPU, const uint resCPU);
}

#endif
