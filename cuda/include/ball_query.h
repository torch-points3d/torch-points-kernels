#pragma once
#include <torch/extension.h>

std::pair<at::Tensor, at::Tensor> ball_query_dense(at::Tensor new_xyz, at::Tensor xyz,
                                                   const float radius, const int nsample);

std::pair<at::Tensor, at::Tensor> ball_query_partial_dense(at::Tensor x, at::Tensor y,
                                                           at::Tensor batch_x, at::Tensor batch_y,
                                                           const float radius, const int nsample);

at::Tensor degree(at::Tensor row, int64_t num_nodes);
