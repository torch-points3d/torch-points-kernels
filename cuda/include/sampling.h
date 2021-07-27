#pragma once
#include <torch/extension.h>

torch::Tensor furthest_point_sampling(torch::Tensor points, const int nsamples);
