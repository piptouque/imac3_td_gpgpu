/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#include <cassert>

namespace IMAC
{
    static constexpr int deviceId = 0;

    __global__ void sumMatricesCUDA(int width, int height,
                                    const int * const dev_a,
                                    const int * const dev_b,
                                    int * const dev_c)
    {
        for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < width; w += blockDim.x * gridDim.x)
        {
            for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < height; h += blockDim.y * gridDim.y)
            {
                int i = h + height * w;
                dev_c[i] = dev_a[i] + dev_b[i];
            }
        }
    }

    void studentJob(int width, int height,
                    const int * const * const a,
                    const int * const * const b,
                    int * const * const c)
	{
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceId));

		ChronoGPU chrGPU;


		// Determining the required number of threads and blocks now,
		// so we can assert the system requirements before any allocation.

		// Choosing the number of threads per block such that it is maximal while
		// abiding to the constraint per dimension.
		unsigned int sqrtMaxThreadsPerBlock = static_cast<unsigned int>(std::floor(std::sqrt(prop.maxThreadsPerBlock)));

		// such that numberThreads.x * threads.y <= prop.maxThreadsPerBlock
		// and also numberThreads.x <= prop.maxThreadsDim[0]
        //          numberThreads.y <= prop.maxThreadsDim[1]

        const dim3 numberThreads = {
                std::min(sqrtMaxThreadsPerBlock, static_cast<unsigned>(prop.maxThreadsDim[0])),
                std::min(sqrtMaxThreadsPerBlock, static_cast<unsigned>(prop.maxThreadsDim[1])),
                1};
        assert(numberThreads.x <= prop.maxThreadsDim[0]);
        assert(numberThreads.y <= prop.maxThreadsDim[1]);
        assert(numberThreads.x * numberThreads.y <= prop.maxThreadsPerBlock);
        const dim3 numberBlocks  = {
                width / numberThreads.x,
                height / numberThreads.y,
                1 };
        assert(numberBlocks.x <= prop.maxGridSize[0]);
        assert(numberBlocks.y <= prop.maxGridSize[1]);


        // 2 arrays (flattened matrices) for GPU
		int * dev_a = nullptr;
        int * dev_b = nullptr;
        int * dev_c = nullptr;

		// alloc, host -> device
        // We will assume column-major.
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_a), width * height * sizeof (int)));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_b), width * height * sizeof (int)));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_c), width * height * sizeof (int)));
        for (int w = 0; w < width; ++w)
        {
            HANDLE_ERROR(cudaMemcpy(dev_a + w * height, a[w], height * sizeof (int), cudaMemcpyHostToDevice));
            HANDLE_ERROR(cudaMemcpy(dev_b + w * height, b[w], height * sizeof (int), cudaMemcpyHostToDevice));
        }


        std::cout << "Matrix addition on GPU (" 	<< numberBlocks.x << "x" << numberBlocks.y << " blocks - "
                  << numberThreads.x << "x" << numberThreads.y << " threads)" << std::endl;
        chrGPU.start();
		sumMatricesCUDA<<<numberBlocks, numberThreads>>>(width, height, dev_a, dev_b, dev_c);
        chrGPU.stop();
        std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        // Starting from (1024, 1024)
        // GPU Computations (flattened) take almost twice the time of CPU computations (multi-dimensional).
        // Below (256, 256)
        // The GPU is slightly faster, but not enough for it to counter-balance the cost of allocations and copies.
        // Verdict: Not worth it??

        // device -> host, dealloc
        for (int w = 0; w < width; ++w)
        {
            HANDLE_ERROR(cudaMemcpy(c[w], dev_c + w * height, height * sizeof (int), cudaMemcpyDeviceToHost));
        }
        HANDLE_ERROR(cudaFree(dev_a));
        HANDLE_ERROR(cudaFree(dev_b));
        HANDLE_ERROR(cudaFree(dev_c));
	}
}
