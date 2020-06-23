#pragma once
#include <torch/extension.h>

#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be a contiguous tensor")
