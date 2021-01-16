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
    __global__ void sepiaCUDA(const int width, const int height,
                              const uchar * const dev_input, uchar * const dev_output);

	// - input: input image RGB
	// - output: output image RGB
    void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output);

}

#endif
