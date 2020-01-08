#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);

std::pair<at::Tensor> ball_query_partial_dense(at::Tensor new_xyz,
					       at::Tensor xyz,
					       at::Tensor batch_new_xyz,
					       at::Tensor batch_xyz,
					       const float radius,
					       const int nsample)
