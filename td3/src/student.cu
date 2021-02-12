/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/
#include <algorithm>
#include "student.hpp"

#include <cuda_runtime_api.h>

namespace IMAC
{
    enum
    {
        KERNEL_EX1 = 0,
        KERNEL_EX2,
        KERNEL_EX3,
        KERNEL_EX4,
        KERNEL_EX5
    };

    __device__
    int cuda_getNumberToProcess(const uint arraySize)
    {
        return blockIdx.x == gridDim.x - 1 ? (arraySize - 1) % (2 * blockDim.x) + 1 : 2 * blockDim.x;
    }


    __device__
    void cuda_fillShrArray(volatile uint* shr_array, const uint* const dev_array, const uint size)
    {
        int shr_idx = 2 * threadIdx.x;
        int dev_idx = shr_idx + 2 * blockIdx.x * blockDim.x;
        if (dev_idx < size)
        {
            // printf("shr_idx: %d, dev_idx: %d \n", shr_idx, dev_idx);
            // printf("%d ", dev_idx);
            shr_array[shr_idx] = dev_array[dev_idx];
            if (dev_idx + 1 < size)
            {
                // printf("shr_idx + 1: %d, dev_idx + 1: %d \n", shr_idx + 1, dev_idx + 1);
                // printf("%d ", dev_idx + 1);
                shr_array[shr_idx + 1] = dev_array[dev_idx + 1];
            }
        }
        __syncthreads();
    }

    // ==================================================== EX 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
        extern __shared__ uint shr_array[];
        // account for unused space in case size is not a power of two.
        int numberToProcess = cuda_getNumberToProcess(size);
        // first, copy the input to the shared memory for this block.
        cuda_fillShrArray(shr_array, dev_array, size);
        // then, in-place max reduction.
        for (int step = 1;; step *= 2)
        {
            int shr_idx = 2 * step * threadIdx.x;
            int shr_next = shr_idx + step;
            if (shr_idx >= numberToProcess || shr_next >= numberToProcess)
            {
                break;
            }
            shr_array[shr_idx] = umax(shr_array[shr_idx], shr_array[shr_next]);
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            // lastly, copy from the shared memory to the corresponding space for this block.
            dev_partialMax[blockIdx.x] = shr_array[0];
        }
        __syncthreads();
    }

    // ==================================================== EX 2, EX3
    __global__
    void maxReduce_ex23(const uint *const dev_array, const uint size, uint *const dev_partialMax)
    {
        extern __shared__ uint shr_array[];
        // account for unused space in case size is not a power of two.
        const int numberToProcess = cuda_getNumberToProcess(size);
        // first, copy the input to the shared memory for this block.
        cuda_fillShrArray(shr_array, dev_array, size);
        // then, in-place max reduction.

        for (int step = 1;; step *= 2)
        {
            int shr_idx = threadIdx.x;
            int numberToProcessStep = (numberToProcess - 1) / (2 * step) + 1;
            int shr_next = shr_idx + numberToProcessStep;
            if (2 * shr_idx >= shr_next || shr_next >= numberToProcess)
            {
                break;
            }
            shr_array[shr_idx] = umax(shr_array[shr_idx], shr_array[shr_next]);
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            // lastly, copy from the shared memory to the corresponding space for this block.
            dev_partialMax[blockIdx.x] = shr_array[0];
        }
        __syncthreads();
    }

    // ==================================================== EX 4
    __global__
    void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax, int warpSize)
    {
        extern __shared__ uint shr_array[];
        volatile uint* shr_array_vol = shr_array;
        // account for unused space in case size is not a power of two.
        const int numberToProcess = cuda_getNumberToProcess(size);
        // first, copy the input to the shared memory for this block.
        cuda_fillShrArray(shr_array_vol, dev_array, size);
        // then, in-place max reduction.
        for (int step = 1;; step *= 2)
        {
            int numberToProcessStep = (numberToProcess - 1) / (2 * step) + 1;
            int shr_idx = threadIdx.x;
            int shr_next = shr_idx + numberToProcessStep;
            if (2 * shr_idx >= shr_next || shr_next >= numberToProcess)
            {
                break;
            }
            shr_array_vol[shr_idx] = umax(shr_array_vol[shr_idx], shr_array_vol[shr_next]);
            if (numberToProcessStep <= warpSize * 2)
            {
                // no need to synchronise if there are only warpSize threads still running.
                __syncthreads();
            }
        }
        if (threadIdx.x == 0)
        {
            // lastly, copy from the shared memory to the corresponding space for this block.
            dev_partialMax[blockIdx.x] = shr_array_vol[0];
        }
        __syncthreads();
    }

    int getWarpSize()
    {
        cudaDeviceProp prop;
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, device));
        return prop.warpSize;
    }

    // return a uint2 with x: dimBlock / y: dimGrid
    template<uint kernelType>
    uint2 configureKernel(const uint sizeArray)
    {
        cudaDeviceProp prop;
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));

        unsigned long maxThreadsPerBlock	= prop.maxThreadsPerBlock;

        // for thread number optimisation.
        const int totalNumberThreads = sizeArray / 2 + 1;

        uint2 dimBlockGrid; // x: dimBlock / y: dimGrid
        // std::cout << "ahhh: " << totalNumberThreads << std::endl;

        // Configure number of threads/blocks
        switch(kernelType)
        {
            case KERNEL_EX1:
            case KERNEL_EX3:
            case KERNEL_EX4:
            case KERNEL_EX2:
                dimBlockGrid.y = (totalNumberThreads - 1) / maxThreadsPerBlock + 1;
                dimBlockGrid.x = (totalNumberThreads - 1) / dimBlockGrid.y + 1;
                /*
                // only allocating a single block of threads if the array length is small.
                dimBlockGrid.x = std::max<uint>(1, std::min<uint>(sizeArray / 2, maxThreadsPerBlock));
                // set number of blocks according to the size of the input array.
                dimBlockGrid.y = std::max<uint>(1, std::max<uint>(1, (sizeArray - 1)) / dimBlockGrid.x);
                 */
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
        std::vector<uint> host_partialMax(dimBlockGrid.y); // REPLACE SIZE !
        const size_t bytesPartialMax = host_partialMax.size() * sizeof(uint); // REPLACE BYTES !
        const size_t bytesSharedMem = 2 * dimBlockGrid.x * sizeof(uint); // REPLACE BYTES !

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
                    maxReduce_ex1<<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax);
                    break;
                case KERNEL_EX2:
                case KERNEL_EX3:
                    // same thing
                    maxReduce_ex23<<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax);
                    break;
                case KERNEL_EX4:
                    /// TODO EX 4
                    maxReduce_ex4<<<dimBlockGrid.y, dimBlockGrid.x, bytesSharedMem>>>(dev_array, size, dev_partialMax, getWarpSize());
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
    void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);

        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

        // in this implementation ex2 and 3 are the same.
        /*
		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);
         */

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);
		
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);
		
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

        /*
		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);
		
        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);
         */

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
