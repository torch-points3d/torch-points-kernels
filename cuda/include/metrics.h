#pragma once
#include <torch/extension.h>

at::Tensor instance_iou_cuda(at::Tensor instance_idx, at::Tensor instance_offsets,
                             at::Tensor gt_instances, at::Tensor gt_instance_sizes,
                             at::Tensor num_gt_instances, at::Tensor batch);
