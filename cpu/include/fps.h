#pragma once
#include <torch/extension.h>
at::Tensor fps(at::Tensor points, const int nsamples, bool random = true);