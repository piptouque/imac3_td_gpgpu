/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

#include <cassert>
#include <cuda_runtime_api.h>

namespace IMAC
{
    static constexpr int deviceId = 0;
    static constexpr int maxConvMatSize = 15;

    __constant__ float cst_matConv[maxConvMatSize * maxConvMatSize];

    texture<uchar4, 1, cudaReadModeElementType> tex1_inputImg;
    texture<uchar4, 2, cudaReadModeElementType> tex2_inputImg;

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
// ==================================================

    int getDeviceMaxTexture1D()
    {
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceId));

        return prop.maxTexture1D;
    }

    int2 getDeviceMaxTexture2D()
    {
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceId));

        return make_int2(prop.maxTexture2D[0], prop.maxTexture2D[1]);
    }

    void computeThreadBlockNumbers(dim3 &numberThreads, dim3 &numberBlocks,
                                   const uint imgWidth, const uint imgHeight
                                   )
    {
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceId));

        // Determining the required number of threads and blocks now,
        // so we can assert the system requirements before any allocation.

        // Choosing the number of threads per block such that it is maximal while
        // abiding to the constraint per dimension.
        unsigned int sqrtMaxThreadsPerBlock = static_cast<unsigned int>(std::floor(std::sqrt(prop.maxThreadsPerBlock)));

        // such that numberThreads.x * threads.y <= prop.maxThreadsPerBlock
        // and also numberThreads.x <= prop.maxThreadsDim[0]
        //          numberThreads.y <= prop.maxThreadsDim[1]

        numberThreads = {
                std::min(sqrtMaxThreadsPerBlock, static_cast<unsigned>(prop.maxThreadsDim[0])),
                std::min(sqrtMaxThreadsPerBlock, static_cast<unsigned>(prop.maxThreadsDim[1])),
                1};
        assert(numberThreads.x <= prop.maxThreadsDim[0]);
        assert(numberThreads.y <= prop.maxThreadsDim[1]);
        assert(numberThreads.x * numberThreads.y <= prop.maxThreadsPerBlock);
        numberBlocks  = {
                imgWidth / numberThreads.x,
                imgHeight / numberThreads.y,
                1 };
        assert(numberBlocks.x <= prop.maxGridSize[0]);
        assert(numberBlocks.y <= prop.maxGridSize[1]);

    }

    __global__ void convCUDA1(const uchar4* dev_inputImg,
                              const uint imgWidth, const uint imgHeight,
                              const float* dev_matConv,
                              const uint matSize,
                              uchar4* dev_output
                              )
    {
	    // The image is column-major.
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
                for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int h1 = 0; h1 < matSize; ++h1)
                {
                    for (int w1 = 0; w1 < matSize; ++w1)
                    {
                        int indexKernel = w1 + matSize * h1;

                        int y = h - matSize / 2 + h1;
                        int x = w - matSize / 2 + w1;
                        // border-padding
                        y = min(imgHeight - 1, max(0, y));
                        x = min(imgWidth  - 1, max(0, x));
                        int indexImg = x + imgWidth * y;

                        const float f = dev_matConv[indexKernel];
                        const uchar4 c = dev_inputImg[indexImg];

                        val.x += static_cast<float>(c.x) * f;
                        val.y += static_cast<float>(c.y) * f;
                        val.z += static_cast<float>(c.z) * f;
                        val.w += static_cast<float>(c.w) * f;
                    }
                }
                const int i = w + imgWidth * h;

                dev_output[i].x = static_cast<uchar>(min(val.x, 255.0f));
                dev_output[i].y = static_cast<uchar>(min(val.y, 255.0f));
                dev_output[i].z = static_cast<uchar>(min(val.z, 255.0f));
                dev_output[i].w = static_cast<uchar>(min(val.w, 255.0f));
            }
        }
    }

    __global__ void convCUDA2(const uchar4* dev_inputImg,
                              const uint imgWidth, const uint imgHeight,
                              const uint matSize,
                              uchar4* dev_output
    )
    {
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
                for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int h1 = 0; h1 < matSize; ++h1)
                {
                    for (int w1 = 0; w1 < matSize; ++w1)
                    {
                        int indexKernel = w1 + matSize * h1;

                        int y = h - matSize / 2 + h1;
                        int x = w - matSize / 2 + w1;
                        // border-padding
                        y = min(imgHeight - 1, max(0, y));
                        x = min(imgWidth  - 1, max(0, x));
                        int indexImg = x + imgWidth * y;

                        const float f = cst_matConv[indexKernel];
                        const uchar4 c = dev_inputImg[indexImg];

                        val.x += static_cast<float>(c.x) * f;
                        val.y += static_cast<float>(c.y) * f;
                        val.z += static_cast<float>(c.z) * f;
                        val.w += static_cast<float>(c.w) * f;
                    }
                }
                const int i = w + imgWidth * h;
                dev_output[i].x = static_cast<uchar>(min(val.x, 255.0f));
                dev_output[i].y = static_cast<uchar>(min(val.y, 255.0f));
                dev_output[i].z = static_cast<uchar>(min(val.z, 255.0f));
                dev_output[i].w = static_cast<uchar>(min(val.w, 255.0f));
            }
        }
    }

    __global__ void convCUDA3(const uint imgWidth, const uint imgHeight,
                              const uint matSize,
                              uchar4* dev_output
    )
    {
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
            for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int h1 = 0; h1 < matSize; ++h1)
                {
                    for (int w1 = 0; w1 < matSize; ++w1)
                    {
                        int indexKernel = w1 + matSize * h1;

                        int y = h - matSize / 2 + h1;
                        int x = w - matSize / 2 + w1;
                        // border-padding
                        x = min(imgWidth  - 1, max(0, x));
                        y = min(imgHeight - 1, max(0, y));
                        const int   i = x + imgWidth * y;

                        const float f = cst_matConv[indexKernel];
                        const uchar4 c = tex1D(tex1_inputImg, i);

                        val.x += static_cast<float>(c.x) * f;
                        val.y += static_cast<float>(c.y) * f;
                        val.z += static_cast<float>(c.z) * f;
                        val.w += static_cast<float>(c.w) * f;
                    }
                }
                const int i = w + imgWidth * h;
                dev_output[i].x = static_cast<uchar>(min(val.x, 255.0f));
                dev_output[i].y = static_cast<uchar>(min(val.y, 255.0f));
                dev_output[i].z = static_cast<uchar>(min(val.z, 255.0f));
                dev_output[i].w = static_cast<uchar>(min(val.w, 255.0f));
                // dev_output[i] = make_uchar4(50, 0, 0, 255);
            }
        }
    }

    __global__ void convCUDA4(const uint imgWidth, const uint imgHeight,
                              const int pitch,
                              const uint matSize,
                              uchar4* dev_output
    )
    {
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
            for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int h1 = 0; h1 < matSize; ++h1)
                {
                    for (int w1 = 0; w1 < matSize; ++w1)
                    {
                        int indexKernel = w1 + matSize * h1;

                        int x = w - matSize / 2 + w1;
                        int y = h - matSize / 2 + h1;
                        // border-padding
                        x = min(imgWidth     - 1, max(0, x));
                        y = min(imgHeight - 1, max(0, y));

                        const float f = cst_matConv[indexKernel];
                        const uchar4 c = tex2D(tex2_inputImg, x, y);

                        val.x += static_cast<float>(c.x) * f;
                        val.y += static_cast<float>(c.y) * f;
                        val.z += static_cast<float>(c.z) * f;
                        val.w += static_cast<float>(c.w) * f;
                    }
                }
                uchar4* dev_output_ptr = reinterpret_cast<uchar4*>(reinterpret_cast<char*>(dev_output) + pitch * h);
                dev_output_ptr[w].x = static_cast<uchar>(min(val.x, 255.0f));
                dev_output_ptr[w].y = static_cast<uchar>(min(val.y, 255.0f));
                dev_output_ptr[w].z = static_cast<uchar>(min(val.z, 255.0f));
                dev_output_ptr[w].w = static_cast<uchar>(min(val.w, 255.0f));
            }
        }
    }

    void exercise1(const std::vector<uchar4> &inputImg, // Input image
                    const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
                    const uint matSize, // Matrix size (width or height)
                    std::vector<uchar4> &output // Output image
    )
    {
	    dim3 numberThreads, numberBlocks;
	    computeThreadBlockNumbers(numberThreads, numberBlocks, imgWidth, imgHeight);

        ChronoGPU chrGPU;

	    //
	    uchar4  *dev_inputImg;
        float   *dev_matConv;
        uchar4   *dev_output;

        // alloc, host -> device
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_inputImg), imgWidth * imgHeight * sizeof(uchar4)));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_matConv), matSize * matSize * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_output), imgWidth * imgHeight * sizeof(uchar4)));

        HANDLE_ERROR(cudaMemcpy(dev_inputImg, inputImg.data(), imgWidth * imgHeight * sizeof(uchar4), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_matConv, matConv.data(), matSize * matSize * sizeof(float), cudaMemcpyHostToDevice));


        std::cout << "Convolution on GPU(" 	<< numberBlocks.x << "x" << numberBlocks.y << " blocks - "
                  << numberThreads.x << "x" << numberThreads.y << " threads)" << std::endl;
        chrGPU.start();
        convCUDA1<<<numberBlocks, numberThreads>>>(dev_inputImg, imgWidth, imgHeight, dev_matConv, matSize, dev_output);
        chrGPU.stop();
        std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

        // device -> host, dealloc
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, imgWidth * imgHeight * sizeof(uchar4), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_inputImg));
        HANDLE_ERROR(cudaFree(dev_matConv));
        HANDLE_ERROR(cudaFree(dev_output));
    }

    void exercise2(const std::vector<uchar4> &inputImg, // Input image
                   const uint imgWidth, const uint imgHeight, // Image size
                   const std::vector<float> &matConv, // Convolution matrix (square)
                   const uint matSize, // Matrix size (width or height)
                   std::vector<uchar4> &output // Output image
    )
    {
        dim3 numberThreads, numberBlocks;
        computeThreadBlockNumbers(numberThreads, numberBlocks, imgWidth, imgHeight);

        ChronoGPU chrGPU;

        //
        uchar4 *dev_inputImg;
        uchar4 *dev_output;

        // alloc, host -> device
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_inputImg), imgWidth * imgHeight * sizeof(uchar4)));
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_output), imgWidth * imgHeight * sizeof(uchar4)));

        HANDLE_ERROR(cudaMemcpy(dev_inputImg, inputImg.data(), imgWidth * imgHeight * sizeof(uchar4), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyToSymbol(cst_matConv, matConv.data(), matSize * matSize * sizeof(float), 0, cudaMemcpyHostToDevice));

        std::cout << "Convolution on GPU, constant kernel (" 	<< numberBlocks.x << "x" << numberBlocks.y << " blocks - "
                  << numberThreads.x << "x" << numberThreads.y << " threads)" << std::endl;

        chrGPU.start();
        convCUDA2<<<numberBlocks, numberThreads>>>(dev_inputImg, imgWidth, imgHeight, matSize, dev_output);
        chrGPU.stop();
        std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

        // device -> host, dealloc
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, imgWidth * imgHeight * sizeof(uchar4), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_inputImg));
        HANDLE_ERROR(cudaFree(dev_output));
    }

    void exercise3(const std::vector<uchar4> &inputImg, // Input image
                   const uint imgWidth, const uint imgHeight, // Image size
                   const std::vector<float> &matConv, // Convolution matrix (square)
                   const uint matSize, // Matrix size (width or height)
                   std::vector<uchar4> &output // Output image
    )
    {
        dim3 numberThreads, numberBlocks;
        computeThreadBlockNumbers(numberThreads, numberBlocks, imgWidth, imgHeight);

        int maxTexture1D = getDeviceMaxTexture1D();

        assert(imgWidth * imgHeight < maxTexture1D && "1-dimensional texture memory is not enough for the image, use 2D.");

        ChronoGPU chrGPU;

        //
        uchar4 *dev_output;

        cudaArray* arr_inputImg;
        // uchar4 -> 8 x 8 x 8 x 8 unsigned.d
        // see: https://stackoverflow.com/a/45047931
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

        // Using 1D texture memory and cudaArray.
        // Quite limited in size, support only up to 2^16 pixels (256 x 256) on my machine.
        const cudaExtent ext = make_cudaExtent(imgWidth * imgHeight, 0, 0);
        // alloc, host -> device
        HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&dev_output), imgWidth * imgHeight * sizeof(uchar4)));
        HANDLE_ERROR(cudaMalloc3DArray(&arr_inputImg, &channelDesc, ext));

        HANDLE_ERROR(cudaMemcpyToSymbol(cst_matConv, matConv.data(), matSize * matSize * sizeof(float), 0, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyToArray(arr_inputImg, 0, 0, inputImg.data(), imgWidth * imgHeight * sizeof(uchar4), cudaMemcpyHostToDevice));

        // texture parameters.
        tex1_inputImg.addressMode[0] = cudaAddressModeClamp;
        tex1_inputImg.normalized = false;

        // bind array to texture
        HANDLE_ERROR(cudaBindTextureToArray(tex1_inputImg, arr_inputImg, channelDesc));

        std::cout << "Convolution on GPU, 1D texture (" 	<< numberBlocks.x << "x" << numberBlocks.y << " blocks - "
                  << numberThreads.x << "x" << numberThreads.y << " threads)" << std::endl;

        chrGPU.start();
        convCUDA3<<<numberBlocks, numberThreads>>>(imgWidth, imgHeight, matSize, dev_output);
        chrGPU.stop();
        std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;


        // device -> host, dealloc
        HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, imgWidth * imgHeight * sizeof(uchar4), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_output));
        HANDLE_ERROR(cudaFreeArray(arr_inputImg));

        HANDLE_ERROR(cudaUnbindTexture(tex1_inputImg));
    }

    void exercise4(const std::vector<uchar4> &inputImg, // Input image
                   const uint imgWidth, const uint imgHeight, // Image size
                   const std::vector<float> &matConv, // Convolution matrix (square)
                   const uint matSize, // Matrix size (width or height)
                   std::vector<uchar4> &output // Output image
    )
    {
        dim3 numberThreads, numberBlocks;
        computeThreadBlockNumbers(numberThreads, numberBlocks, imgWidth, imgHeight);

        int2 maxTexture2D = getDeviceMaxTexture2D();
        assert(imgWidth < maxTexture2D.x && imgHeight < maxTexture2D.y);

        ChronoGPU chrGPU;

        //
        cudaArray* arr_inputImg;
        uchar4 *dev_output;

        // uchar4 -> 8 x 8 x 8 x 8 unsigned.
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

        // Using 2D texture memory and cudaArray.
        const cudaExtent ext = make_cudaExtent(imgWidth, imgHeight, 0);
        // alloc,
        HANDLE_ERROR(cudaMalloc3DArray(&arr_inputImg, &channelDesc, ext));
        std::size_t pitch;
        HANDLE_ERROR(cudaMallocPitch(reinterpret_cast<void **>(&dev_output),
                                     &pitch,
                                     imgWidth * sizeof(uchar4),
                                     imgHeight));
        // host -> device
        HANDLE_ERROR(cudaMemcpyToSymbol(cst_matConv,
                                        matConv.data(),
                                        matSize * matSize * sizeof(float),
                                        0,
                                        cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpyToArray(arr_inputImg,
                                0, 0,
                                inputImg.data(),
                                imgWidth * imgHeight * sizeof(uchar4),
                                cudaMemcpyHostToDevice));

        // texture parameters.
        tex2_inputImg.addressMode[0] = cudaAddressModeClamp;
        tex2_inputImg.addressMode[1] = cudaAddressModeClamp;
        tex2_inputImg.normalized = false;

        // bind array to texture
        HANDLE_ERROR(cudaBindTextureToArray(tex2_inputImg, arr_inputImg, channelDesc));


        std::cout << "Convolution on GPU, 2D texture (" 	<< numberBlocks.x << "x" << numberBlocks.y << " blocks - "
                  << numberThreads.x << "x" << numberThreads.y << " threads)" << std::endl;

        chrGPU.start();
        convCUDA4<<<numberBlocks, numberThreads>>>(imgWidth, imgHeight, pitch, matSize, dev_output);
        chrGPU.stop();
        std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;


        // device -> host, dealloc
        // see: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g3a58270f6775efe56c65ac47843e7cee
        HANDLE_ERROR(cudaMemcpy2D(output.data(),
                                  imgWidth * sizeof(uchar4), // destination pitch
                                  dev_output,
                                  pitch, // source pitch
                                  imgWidth * sizeof(uchar4), // columns in bytes -> row width
                                  imgHeight, // rows
                                  cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_output));
        HANDLE_ERROR(cudaFreeArray(arr_inputImg));

        HANDLE_ERROR(cudaUnbindTexture(tex2_inputImg));
    }

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
        int maxTexture1D = getDeviceMaxTexture1D();
        int2 maxTexture2D = getDeviceMaxTexture2D();



        std::cout << " -- EXERCISE 1 -- " << std::endl;
		exercise1(inputImg, imgWidth, imgHeight, matConv, matSize, output);
        compareImages(resultCPU, output);
        std::cout << " -- EXERCISE 2 -- " << std::endl;
        exercise2(inputImg, imgWidth, imgHeight, matConv, matSize, output);
        compareImages(resultCPU, output);

        // This part can only run with small enough images (on my machine : 256 x 256 or same pixel count tops)
        if (imgWidth * imgHeight < maxTexture1D)
        {
            std::cout << " -- EXERCISE 3 -- " << std::endl;
            exercise3(inputImg, imgWidth, imgHeight, matConv, matSize, output);
            compareImages(resultCPU, output);
        }
        else
        {
            std::cout << " -- WARNING: IMAGE TOO LARGE, CANNOT RUN EXERCISE 3 -- " << std::endl;
        }
        if (imgWidth < maxTexture2D.x && imgHeight < maxTexture2D.y)
        {
            std::cout << " -- EXERCISE 4 -- " << std::endl;
            exercise4(inputImg, imgWidth, imgHeight, matConv, matSize, output);
            compareImages(resultCPU, output);
        }
        else
        {
            std::cout << " -- WARNING: IMAGE TOO LARGE, CANNOT RUN EXERCISE 4 -- " << std::endl;
        }
    }
}
