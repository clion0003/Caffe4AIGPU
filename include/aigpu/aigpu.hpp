#ifndef CONV_H
#define CONV_H

#include <iostream>
#include <string>
template <typename Dtype>
int ConvKernel(std::string name, Dtype* d_out, const Dtype* d_in, const Dtype* d_kernel, const Dtype* d_kernel_bias);

#endif //CONV_H
