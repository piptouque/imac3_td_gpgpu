/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: main.cpp
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include <algorithm>

#include "student.hpp"

#include "chronoCPU.hpp"

namespace IMAC
{
	const int DEFAULT_MATRIX_SIZE = 256;

	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -n <N>: <N> is the size of the matrices (default is "
					<< DEFAULT_MATRIX_SIZE << ")" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

    void sumMatricesCPU(int width, int height,
                                    const int * const * const a,
                                    const int * const * const b,
                                    int * const * const c)
	{
		for (int w = 0; w < width; ++w)
        {
		    for (int h = 0; h < height; ++h)
            {
		        c[w][h] = a[w][h] + b[w][h];
            }
        }
	}

	// Compate two matrices (a and b) of size height, width. Return true if equal
    bool compare(int width, int height,
                        const int * const * const a,
                        const int * const * const b)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int h = 0; h < height; ++h)
            {
                if (a[w][h] != b[w][h])
                {
                    std::cout << "Error at index (" << w << ", " << h << ") : a = " << a[w][h] << " - b = " << b[w][h] << std::endl;
                    return false;
                }

            }
		}
		return true;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
        int width = DEFAULT_MATRIX_SIZE;
        int height = DEFAULT_MATRIX_SIZE;

		// Parse command line
		for (int i = 1; i < argc; ++i ) 
		{
			if (!strcmp( argv[i], "-n")) // Matrix size
			{
				if (sscanf(argv[++i], "%d", &width) != 1)
				{
					printUsageAndExit(argv[0]);
				}
                if (sscanf(argv[++i], "%d", &height) != 1)
                {
                    printUsageAndExit(argv[0]);
                }
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}


		std::cout << "Summing matrices of size (" << width << ", " << height << ")" << std::endl << std::endl;

		ChronoCPU chrCPU;

		// Allocate arrays on CPU
		std::cout 	<< "Allocating input (4 matrices): "
					<< ( ( width * height * 4 * sizeof(int) ) >> 20 ) << " MB on Host" << std::endl;
		chrCPU.start();
		int ** a = new int* [width];
		int ** b = new int* [width];
		int ** resCPU = new int* [width];
        int **resGPU = new int* [width];
		for (int w = 0; w < width; ++w)
        {
		    a[w] = new int[height];
            b[w] = new int[height];
            resCPU[w] = new int[height];
            resGPU[w] = new int[height];
        }
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;

		std::srand(static_cast<unsigned int>(std::time(nullptr)));
		// Init arrays
		for (int w = 0; w < width; ++w)
        {
		    auto gen = [width]() { return std::rand() % width; };
		    std::generate(a[w], a[w] + height, gen);
            std::generate(b[w], b[w] + height, gen);
        }

		// Computation on CPU
        std::cout << "Addition on CPU (sequential)"	<< std::endl;
        chrCPU.start();
		sumMatricesCPU(width, height, a, b, resCPU);
        chrCPU.stop();
        std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;

		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		// Call student's code
		studentJob(width, height, a, b, resGPU);
		
		std::cout << "============================================"	<< std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(width, height, resCPU, resGPU))
		{
			std::cout << " -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}

		// Free memory
        for (int w = 0; w < width; ++w)
        {
            delete[] a[w];
            delete[] b[w];
            delete[] resCPU[w];
            delete[] resGPU[w];
        }
		delete[] a;
		delete[] b;
		delete[] resCPU;
		delete[] resGPU;

	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
