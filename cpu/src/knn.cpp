#include "compat.h"
#include "neighbors.cpp"
#include "neighbors.h"
#include "utils.h"
#include <iostream>
#include <torch/extension.h>

std::pair<at::Tensor, at::Tensor> _single_batch_knn(at::Tensor support, at::Tensor query, int k)
{
    CHECK_CONTIGUOUS(support);
    CHECK_CONTIGUOUS(query);
    if (support.size(0) < k)
        TORCH_CHECK(false,
                    "Not enough points in support to find " + std::to_string(k) + " neighboors")
    std::vector<int64_t> neighbors_indices(query.size(0) * k, -1);
    std::vector<float> neighbors_dists(query.size(0) * k, -1);

    auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    auto options_dist = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    AT_DISPATCH_ALL_TYPES(query.scalar_type(), "knn", [&] {
        auto data_q = query.DATA_PTR<scalar_t>();
        auto data_s = support.DATA_PTR<scalar_t>();
        std::vector<scalar_t> queries_stl =
            std::vector<scalar_t>(data_q, data_q + query.size(0) * query.size(1));
        std::vector<scalar_t> supports_stl =
            std::vector<scalar_t>(data_s, data_s + support.size(0) * support.size(1));

        nanoflann_knn_neighbors<scalar_t>(queries_stl, supports_stl, neighbors_indices,
                                          neighbors_dists, k);
    });
    auto neighbors_dists_ptr = neighbors_dists.data();
    int64_t* neighbors_indices_ptr = neighbors_indices.data();
    auto out = torch::from_blob(neighbors_indices_ptr, {query.size(0), k}, options = options);
    auto out_dists =
        torch::from_blob(neighbors_dists_ptr, {query.size(0), k}, options = options_dist);

    return std::make_pair(out.clone(), out_dists.clone());
}

std::pair<at::Tensor, at::Tensor> dense_knn(at::Tensor support, at::Tensor query, int k)
{
    CHECK_CONTIGUOUS(support);
    CHECK_CONTIGUOUS(query);
    CHECK_CPU(query);
    CHECK_CPU(support);

    int b = query.size(0);
    vector<at::Tensor> batch_idx;
    vector<at::Tensor> batch_dist;
    for (int i = 0; i < b; i++)
    {
        auto out_pair = _single_batch_knn(support[i], query[i], k);
        batch_idx.push_back(out_pair.first);
        batch_dist.push_back(out_pair.second);
    }
    auto out_idx = torch::stack(batch_idx);
    auto out_dist = torch::stack(batch_dist);
    return std::make_pair(out_idx, out_dist);
}