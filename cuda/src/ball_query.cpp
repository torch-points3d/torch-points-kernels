#include "ball_query.h"
#include "compat.h"
#include "utils.h"
#include <torch/extension.h>

std::pair<torch::Tensor, torch::Tensor> query_ball_point_kernel_dense_wrapper(float radius,
                                                                              int nsample,
                                                                              torch::Tensor new_xyz,
                                                                              torch::Tensor xyz);
std::pair<torch::Tensor, torch::Tensor>
query_ball_point_kernel_partial_wrapper(float radius, int nsample, torch::Tensor x, torch::Tensor y,
                                        torch::Tensor batch_x, torch::Tensor batch_y);

std::pair<torch::Tensor, torch::Tensor> ball_query_dense(torch::Tensor new_xyz, torch::Tensor xyz,
                                                         const float radius, const int nsample)
{
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);

    CHECK_CUDA(xyz);
    CHECK_CUDA(new_xyz);

    return query_ball_point_kernel_dense_wrapper(radius, nsample, new_xyz, xyz);
}


std::pair<torch::Tensor, torch::Tensor>
ball_query_partial_dense(torch::Tensor x, torch::Tensor y, torch::Tensor batch_x,
                         torch::Tensor batch_y, const float radius, const int nsample)
{
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(y);
    CHECK_IS_FLOAT(x);
    CHECK_IS_FLOAT(y);
    CHECK_CUDA(x);
    CHECK_CUDA(y);
    CHECK_CUDA(batch_x);
    CHECK_CUDA(batch_y);

    return query_ball_point_kernel_partial_wrapper(radius, nsample, x, y, batch_x, batch_y);
}
