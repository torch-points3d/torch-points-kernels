#include "metrics.h"
#include "compat.h"
#include "utils.h"

void instance_iou_kernel_wrapper(int b, int n, int m, const float* dataset, float* temp, int* idxs);

at::Tensor instance_iou_cuda(at::Tensor instance_idx, at::Tensor instance_offsets,
                             at::Tensor instance_gt)
{
    CHECK_CONTIGUOUS(instance_idx);
    CHECK_CONTIGUOUS(instance_offsets);
    CHECK_CONTIGUOUS(instance_gt);
    CHECK_CUDA(instance_idx)
    CHECK_CUDA(instance_offsets)
    CHECK_CUDA(instance_gt)

    auto num_gt_instances = instance_gt.max(0);
    auto num_proposed_instances = instance_offsets.size(0);
    at::Tensor output =
        torch::zeros({num_proposed_instances, num_gt_instances},
                     at::device(num_gt_instances.device()).dtype(at::ScalarType::Float));

    instance_iou_kernel_wrapper(points.size(0), points.size(1), nsamples, points.DATA_PTR<float>(),
                                tmp.DATA_PTR<float>(), output.DATA_PTR<float>());

    return output;
}
