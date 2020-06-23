#include "metrics.h"
#include "compat.h"
#include "utils.h"

void instance_iou_kernel_wrapper(int nInstance, int nProposal, long* proposals_idx,
                                 long* proposals_offset, long* instance_labels,
                                 long* instance_pointnum, float* proposals_iou);

at::Tensor instance_iou_cuda(at::Tensor instance_idx, at::Tensor instance_offsets,
                             at::Tensor gt_instances, at::Tensor gt_instance_sizes,
                             long num_gt_instances)
{
    CHECK_CONTIGUOUS(instance_idx);
    CHECK_CONTIGUOUS(instance_offsets);
    CHECK_CONTIGUOUS(gt_instances);
    CHECK_CONTIGUOUS(gt_instance_sizes);

    CHECK_CUDA(instance_idx);
    CHECK_CUDA(instance_offsets);
    CHECK_CUDA(gt_instances);
    CHECK_CUDA(gt_instance_sizes);

    long num_proposed_instances = instance_offsets.size(0) - 1;
    at::Tensor output =
        torch::zeros({num_proposed_instances, num_gt_instances},
                     at::device(gt_instances.device()).dtype(at::ScalarType::Float));

    instance_iou_kernel_wrapper(num_gt_instances, num_proposed_instances,
                                instance_idx.DATA_PTR<long>(), instance_offsets.DATA_PTR<long>(),
                                gt_instances.DATA_PTR<long>(), gt_instance_sizes.DATA_PTR<long>(),
                                output.DATA_PTR<float>());

    return output;
}
