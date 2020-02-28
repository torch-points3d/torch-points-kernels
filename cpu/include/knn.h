#pragma once
#include <torch/extension.h>
std::pair<at::Tensor, at::Tensor> dense_knn(at::Tensor support, at::Tensor query, int k);