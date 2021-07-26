#pragma once

#include <torch/extension.h>
#include <vector>




std::vector<torch::Tensor> three_nn(torch::Tensor unknowns, torch::Tensor knows);
torch::Tensor three_interpolate(torch::Tensor points, torch::Tensor idx, torch::Tensor weight);
torch::Tensor three_interpolate_grad(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor weight,
                                  const int m);

std::vector<torch::Tensor> three_nn_kernel_wrapper(torch::Tensor unknown, torch::Tensor known);
torch::Tensor three_interpolate_kernel_wrapper(torch::Tensor points, torch::Tensor idx,
                                               torch::Tensor weight);
torch::Tensor three_interpolate_grad_kernel_wrapper(torch::Tensor grad_out, torch::Tensor idx,
                                                    torch::Tensor weight, const int m);