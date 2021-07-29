#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include <torch/extension.h>
// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
template <typename scalar_t>
__global__ void query_ball_point_kernel_dense(int b, int n, int m, float radius, int nsample,
                                              const scalar_t* __restrict__ new_xyz,
                                              const scalar_t* __restrict__ xyz,
                                              int64_t* __restrict__ idx_out,
                                              scalar_t* __restrict__ dist_out)
{
    int batch_index = blockIdx.x;
    xyz += batch_index * n * 3;
    new_xyz += batch_index * m * 3;
    idx_out += m * nsample * batch_index;
    dist_out += m * nsample * batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    float radius2 = radius * radius;
    for (int j = index; j < m; j += stride)
    {
        scalar_t new_x = new_xyz[j * 3 + 0];
        scalar_t new_y = new_xyz[j * 3 + 1];
        scalar_t new_z = new_xyz[j * 3 + 2];
        for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k)
        {
            scalar_t x = xyz[k * 3 + 0];
            scalar_t y = xyz[k * 3 + 1];
            scalar_t z = xyz[k * 3 + 2];
            scalar_t d2 =
                (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            if (d2 < radius2)
            {
                if (cnt == 0)
                {
                    for (int l = 0; l < nsample; ++l)
                    {
                        idx_out[j * nsample + l] = k;
                    }
                }
                idx_out[j * nsample + cnt] = k;
                dist_out[j * nsample + cnt] = d2;
                ++cnt;
            }
        }
    }
}

template <typename scalar_t>
__global__ void query_ball_point_kernel_partial_dense(int size_x, int size_y, float radius,
                                                      int nsample, const scalar_t* __restrict__ x,
                                                      const scalar_t* __restrict__ y,
                                                      const int64_t* __restrict__ batch_x,
                                                      const int64_t* __restrict__ batch_y,
                                                      int64_t* __restrict__ idx_out,
                                                      scalar_t* __restrict__ dist_out)
{
    // taken from
    // https://github.com/rusty1s/pytorch_cluster/blob/master/cuda/radius_kernel.cu
    const ptrdiff_t batch_idx = blockIdx.x;

    const ptrdiff_t start_idx_x = batch_x[batch_idx];
    const ptrdiff_t end_idx_x = batch_x[batch_idx + 1];

    const ptrdiff_t start_idx_y = batch_y[batch_idx];
    const ptrdiff_t end_idx_y = batch_y[batch_idx + 1];
    float radius2 = radius * radius;

    for (ptrdiff_t n_y = start_idx_y + threadIdx.x; n_y < end_idx_y; n_y += blockDim.x)
    {
        int64_t count = 0;
        for (ptrdiff_t n_x = start_idx_x; n_x < end_idx_x; n_x++)
        {
            scalar_t dist = 0;
            for (ptrdiff_t d = 0; d < 3; d++)
            {
                dist += (x[n_x * 3 + d] - y[n_y * 3 + d]) * (x[n_x * 3 + d] - y[n_y * 3 + d]);
            }
            if (dist <= radius2)
            {
                idx_out[n_y * nsample + count] = n_x;
                dist_out[n_y * nsample + count] = dist;
                count++;
            }
            if (count >= nsample)
            {
                break;
            }
        }
    }
}

std::pair<torch::Tensor, torch::Tensor> query_ball_point_kernel_dense_wrapper(float radius,
                                                                              int nsample,
                                                                              torch::Tensor new_xyz,
                                                                              torch::Tensor xyz)
{
    int b = xyz.size(0);
    int n = xyz.size(1);
    int m = new_xyz.size(1);
    torch::Tensor idx =
        torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample}, torch::CUDA(torch::ScalarType::Long));
    torch::Tensor dist = torch::full({new_xyz.size(0), new_xyz.size(1), nsample}, -1,
                                     torch::CUDA(xyz.scalar_type()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        new_xyz.scalar_type(), "query_ball_point_kernel_dense_cuda",
        (
            [&]
            {
                query_ball_point_kernel_dense<scalar_t><<<b, opt_n_threads(m), 0, stream>>>(
                    b, n, m, radius, nsample, new_xyz.data_ptr<scalar_t>(),
                    xyz.data_ptr<scalar_t>(), idx.data_ptr<int64_t>(), dist.data_ptr<scalar_t>());
            }));

    CUDA_CHECK_ERRORS();
    return std::make_pair(idx, dist);
}

torch::Tensor degree(torch::Tensor row, int64_t num_nodes)
{
    auto zero = torch::zeros(num_nodes, row.options());
    auto one = torch::ones(row.size(0), row.options());
    return zero.scatter_add_(0, row, one);
}

std::pair<torch::Tensor, torch::Tensor>
query_ball_point_kernel_partial_wrapper(float radius, int nsample, torch::Tensor x, torch::Tensor y,
                                        torch::Tensor batch_x, torch::Tensor batch_y)
{

    int size_x = x.size(0);
    int size_y = y.size(0);
    torch::Tensor idx = torch::full({y.size(0), nsample}, -1, torch::CUDA(torch::ScalarType::Long));

    torch::Tensor dist = torch::full({y.size(0), nsample}, -1, torch::CUDA(y.scalar_type()));

    cudaSetDevice(x.get_device());
    auto batch_sizes = (int64_t*)malloc(sizeof(int64_t));
    cudaMemcpy(batch_sizes, batch_x[-1].data_ptr<int64_t>(), sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    auto batch_size = batch_sizes[0] + 1;

    batch_x = degree(batch_x, batch_size);
    batch_x = torch::cat({torch::zeros(1, batch_x.options()), batch_x.cumsum(0)}, 0);
    batch_y = degree(batch_y, batch_size);
    batch_y = torch::cat({torch::zeros(1, batch_y.options()), batch_y.cumsum(0)}, 0);

    // query_ball_point_kernel_partial_dense<<<batch_size, TOTAL_THREADS_SPARSE>>>(
    //     size_x, size_y, radius, nsample, x, y, batch_x, batch_y, idx_out, dist_out);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "query_ball_point_kernel_dense_cuda",
        (
            [&]
            {
                query_ball_point_kernel_partial_dense<scalar_t>
                    <<<batch_size, TOTAL_THREADS_SPARSE>>>(
                        size_x, size_y, radius, nsample, x.data_ptr<scalar_t>(),
                        y.data_ptr<scalar_t>(), batch_x.data_ptr<int64_t>(),
                        batch_y.data_ptr<int64_t>(), idx.data_ptr<int64_t>(),
                        dist.data_ptr<scalar_t>());
            }));

    CUDA_CHECK_ERRORS();

    return std::make_pair(idx, dist);
}
