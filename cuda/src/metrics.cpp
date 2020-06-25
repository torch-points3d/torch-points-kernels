#include "metrics.h"
#include "compat.h"
#include "utils.h"

void instance_iou_kernel_wrapper(long total_gt_instances, long max_gt_instances,
                                 const long* nInstance, int nProposal, const long* proposals_idx,
                                 const long* proposals_offset, const long* instance_labels,
                                 const long* offset_num_gt_instances, const long* batch,
                                 const long* instance_pointnum, float* proposals_iou);

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
    long num_proposed_instances = instance_offsets.size(0) - 1;
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
        total_gt_instances[0], max_gt_instances[0], num_gt_instances.DATA_PTR<long>(),
        num_proposed_instances, instance_idx.DATA_PTR<long>(), instance_offsets.DATA_PTR<long>(),
        gt_instances.DATA_PTR<long>(), offset_num_gt_instances.DATA_PTR<long>(),
        batch.DATA_PTR<long>(), gt_instance_sizes.DATA_PTR<long>(), output.DATA_PTR<float>());

    return output;
}
