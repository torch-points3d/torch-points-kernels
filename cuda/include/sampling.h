#pragma once
#include <torch/extension.h>

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);
