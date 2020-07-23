#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel_dense(int b, int n, int m, float radius, int nsample,
                                              const float* __restrict__ new_xyz,
                                              const float* __restrict__ xyz,
                                              int64_t* __restrict__ idx_out,
                                              float* __restrict__ dist_out)
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
        float new_x = new_xyz[j * 3 + 0];
        float new_y = new_xyz[j * 3 + 1];
        float new_z = new_xyz[j * 3 + 2];
        for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k)
        {
            float x = xyz[k * 3 + 0];
            float y = xyz[k * 3 + 1];
            float z = xyz[k * 3 + 2];
            float d2 =
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

__global__ void query_ball_point_kernel_partial_dense(int size_x, int size_y, float radius,
                                                      int nsample, const float* __restrict__ x,
                                                      const float* __restrict__ y,
                                                      const int64_t* __restrict__ batch_x,
                                                      const int64_t* __restrict__ batch_y,
                                                      int64_t* __restrict__ idx_out,
                                                      float* __restrict__ dist_out)
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
            float dist = 0;
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

void query_ball_point_kernel_dense_wrapper(int b, int n, int m, float radius, int nsample,
                                           const float* new_xyz, const float* xyz, int64_t* idx,
                                           float* dist_out)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    query_ball_point_kernel_dense<<<b, opt_n_threads(m), 0, stream>>>(b, n, m, radius, nsample,
                                                                      new_xyz, xyz, idx, dist_out);

    CUDA_CHECK_ERRORS();
}

void query_ball_point_kernel_partial_wrapper(int64_t batch_size, int size_x, int size_y,
                                             float radius, int nsample, const float* x,
                                             const float* y, const int64_t* batch_x,
                                             const int64_t* batch_y, int64_t* idx_out,
                                             float* dist_out)
{
    query_ball_point_kernel_partial_dense<<<batch_size, TOTAL_THREADS_SPARSE>>>(
        size_x, size_y, radius, nsample, x, y, batch_x, batch_y, idx_out, dist_out);

    CUDA_CHECK_ERRORS();
}
