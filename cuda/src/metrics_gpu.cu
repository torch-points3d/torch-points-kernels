#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

#define THREADS 512

__global__ void instance_iou_cuda_kernel(int nInstance, int nProposal, long* proposals_idx,
                                         long* proposals_offset, long* instance_labels,
                                         long* instance_pointnum, float* proposals_iou)
{
    for (int proposal_id = blockIdx.x; proposal_id < nProposal; proposal_id += gridDim.x)
    {
        int start = proposals_offset[proposal_id];
        int end = proposals_offset[proposal_id + 1];
        int proposal_total = end - start;
        for (int instance_id = threadIdx.x; instance_id < nInstance; instance_id += blockDim.x)
        {
            int instance_total = instance_pointnum[instance_id];
            int intersection = 0;
            for (int i = start; i < end; i++)
            {
                int idx = proposals_idx[i];
                if ((int)instance_labels[idx] == instance_id + 1)
                { // 0 is reserved for "no instance"
                    intersection += 1;
                }
            }
            proposals_iou[proposal_id * nInstance + instance_id] =
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
void instance_iou_kernel_wrapper(int nInstance, int nProposal, long* proposals_idx,
                                 long* proposals_offset, long* instance_labels,
                                 long* instance_pointnum, float* proposals_iou)
{
    instance_iou_cuda_kernel<<<std::min(nProposal, THREADS * THREADS),
                               std::min(nInstance, THREADS)>>>(nInstance, nProposal, proposals_idx,
                                                               proposals_offset, instance_labels,
                                                               instance_pointnum, proposals_iou);
}
