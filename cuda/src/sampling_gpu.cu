#include "cuda_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

template <typename scalar_t>
__device__ void __update(scalar_t* __restrict__ dists, int* __restrict__ dists_i, int idx1,
                         int idx2)
{
    const scalar_t v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <typename scalar_t, unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m,
                                               const scalar_t* __restrict__ dataset,
                                               scalar_t* __restrict__ temp, int* __restrict__ idxs)
{
    if (m <= 0)
        return;
    __shared__ scalar_t dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
        idxs[0] = old;

    __syncthreads();
    for (int j = 0; j < m; j++)
    {
        int besti = 0;
        scalar_t best = -1;
        scalar_t x1 = dataset[old * 3 + 0];
        scalar_t y1 = dataset[old * 3 + 1];
        scalar_t z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride)
        {
            scalar_t x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            scalar_t mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            if (mag <= 1e-3)
                continue;

            scalar_t d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

            scalar_t d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 512)
        {
            if (tid < 256)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256)
        {
            if (tid < 128)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128)
        {
            if (tid < 64)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64)
        {
            if (tid < 32)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32)
        {
            if (tid < 16)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16)
        {
            if (tid < 8)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8)
        {
            if (tid < 4)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4)
        {
            if (tid < 2)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2)
        {
            if (tid < 1)
            {
                __update<scalar_t>(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            idxs[j] = old;
    }
}

torch::Tensor furthest_point_sampling_kernel_wrapper(torch::Tensor points, const int nsamples)
{

    int b = points.size(0);
    int n = points.size(1);
    int m = nsamples;
    torch::Tensor idxs =
        torch::zeros({points.size(0), nsamples}, torch::CUDA(torch::ScalarType::Int));

    float init_num = 0;
    switch (points.scalar_type())
    {
    case torch::ScalarType::Half:
        init_num = 65504;
        break;
    default:
        init_num = 1e10;
        break;
    }

    torch::Tensor temp =
        torch::full({points.size(0), points.size(1)}, init_num, torch::CUDA(points.scalar_type()));
    const unsigned int n_threads = opt_n_threads(n);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // switch (n_threads)
    // {
    // case 512:
    //     furthest_point_sampling_kernel<512>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 256:
    //     furthest_point_sampling_kernel<256>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 128:
    //     furthest_point_sampling_kernel<128>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 64:
    //     furthest_point_sampling_kernel<64>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 32:
    //     furthest_point_sampling_kernel<32>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 16:
    //     furthest_point_sampling_kernel<16>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 8:
    //     furthest_point_sampling_kernel<8>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 4:
    //     furthest_point_sampling_kernel<4>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 2:
    //     furthest_point_sampling_kernel<2>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // case 1:
    //     furthest_point_sampling_kernel<1>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    //     break;
    // default:
    //     furthest_point_sampling_kernel<512>
    //         <<<b, n_threads, 0, stream>>>(b, n, m, points, temp, idxs);
    // }

    switch (n_threads)
    {
    case 512:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 512><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 256:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 256><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 128:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 128><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 64:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 64><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 32:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 32><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 16:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 16><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 8:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 8><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 4:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 4><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 2:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 2><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    case 1:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 1><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    default:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            points.scalar_type(), "furthest_point_sampling_kernel_cuda",
            (
                [&]
                {
                    furthest_point_sampling_kernel<scalar_t, 512><<<b, n_threads, 0, stream>>>(
                        b, n, m, points.data_ptr<scalar_t>(), temp.data_ptr<scalar_t>(),
                        idxs.data_ptr<int>());
                }));
        break;
    }

    CUDA_CHECK_ERRORS();
    return idxs;
}
