import torch
from typing import List, Optional
import numpy as np
import numba

if torch.cuda.is_available():
    import torch_points_kernels.points_cuda as tpcuda


@numba.jit(nopython=True, parallel=True)
def _instance_iou_cpu(
    instance_idx,
    instance_offsets,
    gt_instances,
    gt_instance_sizes,
    num_gt_instances: np.array,
    batch: np.array,
):
    num_proposed_instances = len(instance_offsets) - 1
    iou = np.zeros((num_proposed_instances, num_gt_instances.sum()))
    offset_num_gt_instances = np.concatenate((np.array([0]), num_gt_instances.cumsum()))
    for proposed_instance in range(num_proposed_instances):
        instance = instance_idx[instance_offsets[proposed_instance] : instance_offsets[proposed_instance + 1]]
        sample_idx = batch[instance[0]]
        gt_count_offset = offset_num_gt_instances[sample_idx]
        sample_instance_count = num_gt_instances[sample_idx]
        for instance_id in numba.prange(1, sample_instance_count + 1):
            intersection = 0
            for idx in instance:
                if gt_instances[idx] == instance_id:
                    intersection += 1
            iou[proposed_instance, gt_count_offset + instance_id - 1] = intersection / float(
                len(instance) + gt_instance_sizes[gt_count_offset + instance_id - 1] - intersection
            )
    return iou


def instance_iou(
    instance_idx: List[torch.Tensor],
    gt_instances: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
):
    """Computes the IoU between each proposed instance in instance_idx and ground truth instances. Returns a
    tensor of shape [instance_idx.shape[0], num_instances] that contains the iou between the proposed instances and all gt instances
    Instance label 0 is reserved for non instance points

    Parameters
    ----------
    instance_idx : List[torch.Tensor]
        List of instances. Each tensor in this list is a proposed and contains the index of the points
        that belong to that particular instance
    gt_instances : torch.Tensor
        Ground truth instances, contains the index of the instance for each point

    Returns
    -------
    ious: torch.Tensor[nb_proposals, nb_groundtruth]
    """
    if batch is None:
        batch = torch.zeros_like(gt_instances)

    # Gather number of gt instances per batch and size of those instances
    gt_instance_sizes = []
    num_gt_instances = []
    batch_size = batch[-1] + 1
    for s in range(batch_size):
        batch_mask = batch == s
        sample_gt_instances = gt_instances[batch_mask]
        sample_num_gt_instances = torch.max(sample_gt_instances).item()
        num_gt_instances.append(sample_num_gt_instances)
        for instance_id in range(1, sample_num_gt_instances + 1):
            gt_instance_sizes.append(torch.sum(sample_gt_instances == instance_id))
    gt_instance_sizes = torch.stack(gt_instance_sizes)
    num_gt_instances = torch.tensor(num_gt_instances)

    # Instance offset when flatten
    instance_offsets = [0]
    cum_offset = 0
    for instance in instance_idx:
        cum_offset += instance.shape[0]
        instance_offsets.append(cum_offset)

    # Compute ious
    instance_idx = torch.cat(instance_idx)
    if gt_instances.is_cuda:
        return tpcuda.instance_iou_cuda(
            instance_idx.cuda(),
            torch.tensor(instance_offsets).cuda(),
            gt_instances.cuda(),
            gt_instance_sizes.cuda(),
            num_gt_instances.cuda(),
            batch.cuda(),
        )
    else:
        res = _instance_iou_cpu(
            instance_idx.numpy(),
            np.asarray(instance_offsets),
            gt_instances.numpy(),
            gt_instance_sizes.numpy(),
            num_gt_instances.numpy(),
            batch.numpy(),
        )
        return torch.tensor(res).float()
