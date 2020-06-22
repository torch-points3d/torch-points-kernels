#pragma once
#include <torch/extension.h>

at::Tensor instance_iou_cuda(at::Tensor instance_idx, at::Tensor instance_offsets,
                             at::Tensor instance_gt);