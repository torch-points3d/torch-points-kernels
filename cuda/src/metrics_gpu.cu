#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

#define THREADS 512

__global__ void instance_iou_cuda_kernel(
    long total_gt_instances, const long* __restrict__ nInstance, int nProposal,
    const long* __restrict__ proposals_idx, const long* __restrict__ proposals_offset,
    const long* __restrict__ instance_labels, const long* __restrict__ offset_num_gt_instances,
    const long* __restrict__ batch, const long* __restrict__ instance_pointnum,
    float* proposals_iou)
{
    for (int proposal_id = blockIdx.x; proposal_id < nProposal; proposal_id += gridDim.x)
    {
        int start = proposals_offset[proposal_id];
        int end = proposals_offset[proposal_id + 1];
        int sampleIdx = batch[proposals_idx[start]];
        int sampleNInstances = nInstance[sampleIdx];
        int instanceOffset = offset_num_gt_instances[sampleIdx];
        int proposal_total = end - start;
        for (int instance_id = threadIdx.x; instance_id < sampleNInstances;
             instance_id += blockDim.x)
        {
            int instance_total = instance_pointnum[instanceOffset + instance_id];
            int intersection = 0;
            for (int i = start; i < end; i++)
            {
                int idx = proposals_idx[i];
                if ((int)instance_labels[idx] == instance_id + 1)
                { // 0 is reserved for "no instance"
                    intersection += 1;
                }
            }

            proposals_iou[instanceOffset + instance_id + proposal_id * total_gt_instances] =
                (float)intersection /
                ((float)(proposal_total + instance_total - intersection) + 1e-5);
        }
    }
}

// input: proposals_idx (sumNPoint), int
// input: proposals_offset (nProposal + 1), int
// input: instance_labels (N), long, 0~total_nInst-1, -100
// input: instance_pointnum (total_nInst), int
// output: proposals_iou (nProposal, total_nInst), float
void instance_iou_kernel_wrapper(long total_gt_instances, long max_gt_instances,
                                 const long* nInstance, int nProposal, const long* proposals_idx,
                                 const long* proposals_offset, const long* instance_labels,
                                 const long* offset_num_gt_instances, const long* batch,
                                 const long* instance_pointnum, float* proposals_iou)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    instance_iou_cuda_kernel<<<std::min(nProposal, THREADS * THREADS),
                               std::min(max_gt_instances, (long)THREADS), 0, stream>>>(
        total_gt_instances, nInstance, nProposal, proposals_idx, proposals_offset, instance_labels,
        offset_num_gt_instances, batch, instance_pointnum, proposals_iou);
}
