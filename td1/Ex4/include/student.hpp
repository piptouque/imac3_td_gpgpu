/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.hpp
* Author: Maxime MARIA
*/


#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"

namespace IMAC
{
	// Kernel:
	// TODO

	// - input: input flattened matrices a, b
	// - output: output flattened matrix c = a + b
    __global__ void sumMatricesCUDA(int width, int height,
                                    const int * const dev_a,
                                    const int * const dev_b,
                                    int * const dev_c);

    void studentJob(int width, int height,
                    const int * const * const a,
                    const int * const * const b,
                    int * const * const c);

}

#endif
