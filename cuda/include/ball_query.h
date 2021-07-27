#pragma once
#include <torch/extension.h>

std::pair<torch::Tensor, torch::Tensor> ball_query_dense(torch::Tensor new_xyz, torch::Tensor xyz,
                                                         const float radius, const int nsample);

std::pair<torch::Tensor, torch::Tensor>
ball_query_partial_dense(torch::Tensor x, torch::Tensor y, torch::Tensor batch_x,
                         torch::Tensor batch_y, const float radius, const int nsample);

torch::Tensor degree(torch::Tensor row, int64_t num_nodes);
