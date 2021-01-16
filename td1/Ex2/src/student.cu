/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include <cassert>
#include <cmath>
#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
  static constexpr int dimId = 0;
  static constexpr int deviceId = 0;


	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
	    long long int totalNumberThreads = blockDim.x * gridDim.x;
	    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += totalNumberThreads)
        {
            dev_res[i] = dev_a[i] + dev_b[i];
        }
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    
		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);

		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
        HANDLE_ERROR(cudaMalloc((void **)&dev_a, bytes));
        HANDLE_ERROR(cudaMalloc((void **)&dev_b, bytes));
        HANDLE_ERROR(cudaMalloc((void **)&dev_res, bytes));
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
        HANDLE_ERROR(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice));

        const int numberBlocks  = size / prop.maxThreadsDim[dimId] + 1;
        assert(numberBlocks <= prop.maxGridSize[dimId]);
        const int numberThreads =  numberBlocks > 1 ? prop.maxThreadsDim[dimId] : size;
        assert(numberThreads <= prop.maxThreadsPerBlock);

		// Launch kernel
        std::cout << "Addition on GPU (" << numberBlocks << " blocks - " << numberThreads << " threads)" << std::endl;
        chrGPU.start();
        sumArraysCUDA<<<numberBlocks, numberThreads>>>(size, dev_a, dev_b, dev_res);
        chrGPU.stop();
        std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl;

		// Copy data from device to host (output array)  
        HANDLE_ERROR(cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost));

		// Free arrays on device
        HANDLE_ERROR(cudaFree(dev_a));
        HANDLE_ERROR(cudaFree(dev_b));
        HANDLE_ERROR(cudaFree(dev_res));
	}
}

