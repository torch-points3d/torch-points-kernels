import torch
from typing import List
import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def _instance_iou_cpu(instance_idx, instance_offsets, instance_gt):
    num_instances = np.max(instance_gt)
    gt_instance_sizes = []
    for instance_id in numba.prange(1, num_instances + 1):
        gt_instance_sizes.append(np.sum(instance_gt == instance_id))
    iou = np.zeros((len(instance_offsets), num_instances))
    old_offset = 0
    for proposed_instance, offset in enumerate(instance_offsets):
        instance = instance_idx[old_offset:offset]
        for instance_id in numba.prange(1, num_instances + 1):
            intersection = 0
            for idx in instance:
                if instance_gt[idx] == instance_id:
                    intersection += 1
            iou[proposed_instance, instance_id - 1] = intersection / float(
                len(instance) + gt_instance_sizes[instance_id - 1] - intersection
            )
        old_offset = offset
    return iou


def instance_iou(instance_idx: List[torch.Tensor], gt_instances: torch.Tensor):
    """ Computes the IoU between each proposed instance in instance_idx and ground truth instances. Returns a 
    tensor of shape [instance_idx.shape[0], num_instances] that contains the iou between the proposed instances and all gt instances
    Instance label 0 is reserved for non instance points

    Parameters
    ----------
    instance_idx : List[torch.Tensor]
        List of instances. Each tensor in this list is a proposed and contains the index of the points 
        that belong to that particular instance
    gt_instances : torch.Tensor
        Ground truth instances, contains the index of the instance for each point
    num_instances : int
        Number of instances in ground truth

    Returns
    -------
    ious: 
    """
    instance_offsets = []
    cum_offset = 0
    for instance in instance_idx:
        cum_offset += instance.shape[0]
        instance_offsets.append(cum_offset)
    instance_idx = torch.cat(instance_idx)
    if gt_instances.is_cuda:
        pass
    else:
        res = _instance_iou_cpu(
            instance_idx.numpy(), np.asarray(instance_offsets), gt_instances.numpy()
        )
        return torch.tensor(res).float()
