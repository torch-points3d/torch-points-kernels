#include "gridding.h"
#include "utils.h"

std::vector<torch::Tensor> gridding(float min_x, float max_x, float min_y, float max_y, float min_z,
                                    float max_z, torch::Tensor ptcloud)
{
    CHECK_CUDA(ptcloud);
    CHECK_CONTIGUOUS(ptcloud);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return gridding_kernel_warpper(min_x, max_x, min_y, max_y, min_z, max_z, ptcloud, stream);
}

torch::Tensor gridding_grad(torch::Tensor grid_pt_weights, torch::Tensor grid_pt_indexes,
                            torch::Tensor grad_grid)
{
    CHECK_CUDA(grid_pt_weights);
    CHECK_CONTIGUOUS(grid_pt_weights);
    CHECK_CUDA(grid_pt_indexes);
    CHECK_CONTIGUOUS(grid_pt_indexes);
    CHECK_CUDA(grad_grid);
    CHECK_CONTIGUOUS(grad_grid);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return gridding_grad_kernel_warpper(grid_pt_weights, grid_pt_indexes, grad_grid, stream);
}
