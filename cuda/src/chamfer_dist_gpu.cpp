#include "chamfer_dist.h"

std::vector<torch::Tensor> chamfer_dist(torch::Tensor xyz1, torch::Tensor xyz2)
{
    return chamfer_dist_kernel_wrapper(xyz1, xyz2);
}

std::vector<torch::Tensor> chamfer_dist_grad(torch::Tensor xyz1, torch::Tensor xyz2,
                                             torch::Tensor idx1, torch::Tensor idx2,
                                             torch::Tensor grad_dist1, torch::Tensor grad_dist2)
{
    return chamfer_dist_grad_kernel_wrapper(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2);
}
