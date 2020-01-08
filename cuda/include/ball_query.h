#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);

at::Tensor ball_query_shifted(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);