#include "cubic_feature_sampling.h"
#include "utils.h"

std::vector<torch::Tensor> cubic_feature_sampling(int scale, int neighborhood_size,
                                                  torch::Tensor ptcloud,
                                                  torch::Tensor cubic_features)
{
    CHECK_CUDA(ptcloud);
    CHECK_CONTIGUOUS(ptcloud);
    CHECK_CUDA(cubic_features);
    CHECK_CONTIGUOUS(cubic_features);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return cubic_feature_sampling_kernel_wrapper(scale, neighborhood_size, ptcloud, cubic_features,
                                                 stream);
}

std::vector<torch::Tensor> cubic_feature_sampling_grad(int scale, int neighborhood_size,
                                                       torch::Tensor grad_point_features,
                                                       torch::Tensor grid_pt_indexes)
{
    CHECK_CUDA(grad_point_features);
    CHECK_CONTIGUOUS(grad_point_features);
    CHECK_CUDA(grid_pt_indexes);
    CHECK_CONTIGUOUS(grid_pt_indexes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return cubic_feature_sampling_grad_kernel_wrapper(scale, neighborhood_size, grad_point_features,
                                                      grid_pt_indexes, stream);
}
