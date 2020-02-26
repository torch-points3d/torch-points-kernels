#pragma once
#include <torch/extension.h>
std::pair<at::Tensor, at::Tensor> dense_knn(at::Tensor query, at::Tensor support, int k);