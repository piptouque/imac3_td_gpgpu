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

    __global__ void sepiaCUDA(const int width, const int height, const uchar * const dev_input, uchar * const dev_output)
    {
        // I could not notice any difference between column-major and row-major orders.
        // If there is a difference, it could be a result of a gpu's internal architecture.
        for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < width; w += blockDim.x * gridDim.x)
        {
            for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < height; h += blockDim.y * gridDim.y)
            {
                size_t i = 3 * (w + width * h);
                int r = dev_input[i];
                int g = dev_input[i + 1];
                int b = dev_input[i + 2];
                dev_output[i]     = fminf((uchar)255, r * 0.393 + g * 0.769 + b * 0.189);
                dev_output[i + 1] = fminf((uchar)255, r * 0.349 + g * 0.686 + b * 0.168);
                dev_output[i + 2] = fminf((uchar)255, r * 0.272 + g * 0.534 + b * 0.131);
            }
        }
    }

    void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
        assert(3 * width * height == input.size());

        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceId));

		ChronoGPU chrGPU;

		// Determining the required number of threads and blocks now,
		// so we can assert the system requirements before any allocation.

		// Choosing the number of threads per block such that it is maximal while
		// abiding to the constraint per dimension.
		auto sqrtMaxThreadsPerBlock = static_cast<unsigned int>(std::floor(std::sqrt(prop.maxThreadsPerBlock)));

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


        // 2 arrays for GPU
		uchar *dev_input = nullptr;
        uchar *dev_output = nullptr;

		const std::size_t numberBytes = input.size() * sizeof(uchar);

		// alloc
		HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_input), numberBytes));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_output), numberBytes));

		// host -> device
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), numberBytes, cudaMemcpyHostToDevice));


        chrGPU.start();
        std::cout << "Sepia filter on GPU (" 	<< numberBlocks.x << "x" << numberBlocks.y << " blocks - "
                  << numberThreads.x << "x" << numberThreads.y << " threads)" << std::endl;
		sepiaCUDA<<<numberBlocks, numberThreads>>>(width, height, dev_input, dev_output);
        chrGPU.stop();
        std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
        // Comparing between three images of different resolutions.
        // Computations are consistently around 10 times faster on the GPU.

        // device -> host
        output.reserve(input.size());
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, numberBytes, cudaMemcpyDeviceToHost));

        // dealloc
		HANDLE_ERROR(cudaFree(dev_input));
        HANDLE_ERROR(cudaFree(dev_output));
	}
}
