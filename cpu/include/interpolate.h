#pragma once
#include <torch/extension.h>

at::Tensor knn_interpolate(at::Tensor features, at::Tensor idx, at::Tensor weight);

at::Tensor knn_interpolate_grad(at::Tensor grad_out, at::Tensor idx, at::Tensor weight,
                                const int m);