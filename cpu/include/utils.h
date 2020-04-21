#pragma once
#include <torch/extension.h>

#define CHECK_CPU(x) TORCH_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_INT_TENSOR(x)                                                                        \
    TORCH_CHECK(x.dtype() == torch::kUInt8 || x.dtype() == torch::kInt8 ||                         \
                    x.dtype() == torch::kInt16 || x.dtype() == torch::kInt32 ||                    \
                    x.dtype() == torch::kInt64,                                                    \
                #x " must be of integer type")

#define CHECK_FLOAT_TENSOR(x)                                                                      \
    TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32,                      \
                #x " must be of floating type")
