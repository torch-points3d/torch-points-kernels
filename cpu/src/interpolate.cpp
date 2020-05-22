#include "compat.h"
#include "utils.h"
#include <iostream>
#include <torch/extension.h>

at::Tensor knn_interpolate(at::Tensor features, at::Tensor idx, at::Tensor weight)
{
    CHECK_CONTIGUOUS(features);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_CPU(idx);
    CHECK_CPU(features);
    CHECK_CPU(weight);

    at::Tensor output = torch::zeros({features.size(0), features.size(1), idx.size(1)},
                                     at::device(features.device()).dtype(features.scalar_type()));

    AT_DISPATCH_ALL_TYPES(features.scalar_type(), "knn_interpolate", [&] {
        auto output_a = output.accessor<scalar_t, 3>();
        auto features_a = features.accessor<scalar_t, 3>();
        auto weight_a = weight.accessor<scalar_t, 3>();
        auto idx_a = idx.accessor<int64_t, 3>();

        auto batch_size = idx.size(0);
        for (auto b = 0; b < batch_size; b++)
        {
            for (auto p = 0; p < idx.size(1); p++)
            {
                for (auto c = 0; c < features.size(1); c++)
                {
                    output_a[b][c][p] = 0;
                    for (int i = 0; i < idx.size(2); i++)
                    {
                        auto new_idx = idx_a[b][p][i];
                        output_a[b][c][p] += features_a[b][c][new_idx] * weight_a[b][p][i];
                    }
                }
            }
        }
    });
    return output;
}

at::Tensor knn_interpolate_grad(at::Tensor grad_out, at::Tensor idx, at::Tensor weight, const int m)
{
    CHECK_CPU(grad_out);
    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), m},
                                     at::device(grad_out.device()).dtype(grad_out.scalar_type()));

    AT_DISPATCH_ALL_TYPES(grad_out.scalar_type(), "knn_interpolate_grad", [&] {
        auto output_a = output.accessor<scalar_t, 3>();
        auto grad_out_a = grad_out.accessor<scalar_t, 3>();
        auto weight_a = weight.accessor<scalar_t, 3>();
        auto idx_a = idx.accessor<int64_t, 3>();

        auto batch_size = idx.size(0);
        for (auto b = 0; b < batch_size; b++)
        {
            for (auto p = 0; p < idx.size(1); p++)
            {
                for (auto c = 0; c < grad_out.size(1); c++)
                {
                    for (int i = 0; i < idx.size(2); i++)
                    {
                        auto new_idx = idx_a[b][p][i];
                        output_a[b][c][new_idx] += grad_out_a[b][c][p] * weight_a[b][p][i];
                    }
                }
            }
        }
    });
    return output;
}
