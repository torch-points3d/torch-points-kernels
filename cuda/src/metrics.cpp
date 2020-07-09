#include "metrics.h"
#include "compat.h"
#include "utils.h"

void instance_iou_kernel_wrapper(int64_t total_gt_instances, int64_t max_gt_instances,
                                 const int64_t* nInstance, int nProposal,
                                 const int64_t* proposals_idx, const int64_t* proposals_offset,
                                 const int64_t* instance_labels,
                                 const int64_t* offset_num_gt_instances, const int64_t* batch,
                                 const int64_t* instance_pointnum, float* proposals_iou);

at::Tensor instance_iou_cuda(at::Tensor instance_idx, at::Tensor instance_offsets,
                             at::Tensor gt_instances, at::Tensor gt_instance_sizes,
                             at::Tensor num_gt_instances, at::Tensor batch)
{
    CHECK_CONTIGUOUS(instance_idx);
    CHECK_CONTIGUOUS(instance_offsets);
    CHECK_CONTIGUOUS(gt_instances);
    CHECK_CONTIGUOUS(gt_instance_sizes);
    CHECK_CONTIGUOUS(num_gt_instances);
    CHECK_CONTIGUOUS(batch);

    CHECK_CUDA(instance_idx);
    CHECK_CUDA(instance_offsets);
    CHECK_CUDA(gt_instances);
    CHECK_CUDA(gt_instance_sizes);

    cudaSetDevice(instance_idx.get_device());
    int64_t num_proposed_instances = instance_offsets.size(0) - 1;
    auto total_gt_instances = (int64_t*)malloc(sizeof(int64_t));
    cudaMemcpy(total_gt_instances, num_gt_instances.sum().DATA_PTR<int64_t>(), sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    auto max_gt_instances = (int64_t*)malloc(sizeof(int64_t));
    cudaMemcpy(max_gt_instances, num_gt_instances.max().DATA_PTR<int64_t>(), sizeof(int64_t),
               cudaMemcpyDeviceToHost);

    at::Tensor output =
        torch::zeros({num_proposed_instances, total_gt_instances[0]},
                     at::device(gt_instances.device()).dtype(at::ScalarType::Float));

    at::Tensor offset_num_gt_instances =
        at::cat({at::zeros(1, num_gt_instances.options()), num_gt_instances.cumsum(0)}, 0);
    instance_iou_kernel_wrapper(
        total_gt_instances[0], max_gt_instances[0], num_gt_instances.DATA_PTR<int64_t>(),
        num_proposed_instances, instance_idx.DATA_PTR<int64_t>(),
        instance_offsets.DATA_PTR<int64_t>(), gt_instances.DATA_PTR<int64_t>(),
        offset_num_gt_instances.DATA_PTR<int64_t>(), batch.DATA_PTR<int64_t>(),
        gt_instance_sizes.DATA_PTR<int64_t>(), output.DATA_PTR<float>());

    return output;
}
