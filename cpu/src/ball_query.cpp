
#include "ball_query.h"
#include "compat.h"
#include "neighbors.cpp"
#include "neighbors.h"
#include "utils.h"
#include <iostream>
#include <torch/extension.h>

std::pair<at::Tensor, at::Tensor> ball_query(at::Tensor support, at::Tensor query, float radius,
                                             int max_num, int mode, bool sorted)
{
    CHECK_CONTIGUOUS(support);
    CHECK_CONTIGUOUS(query);

    at::Tensor out;
    at::Tensor out_dists;
    std::vector<int64_t> neighbors_indices(query.size(0), 0);
    std::vector<float> neighbors_dists(query.size(0), -1);

    auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    auto options_dist = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    int max_count = 0;

    AT_DISPATCH_ALL_TYPES(query.scalar_type(), "radius_search", [&] {
        auto data_q = query.DATA_PTR<scalar_t>();
        auto data_s = support.DATA_PTR<scalar_t>();
        std::vector<scalar_t> queries_stl =
            std::vector<scalar_t>(data_q, data_q + query.size(0) * query.size(1));
        std::vector<scalar_t> supports_stl =
            std::vector<scalar_t>(data_s, data_s + support.size(0) * support.size(1));

        max_count = nanoflann_neighbors<scalar_t>(queries_stl, supports_stl, neighbors_indices,
                                                  neighbors_dists, radius, max_num, mode, sorted);
    });
    auto neighbors_dists_ptr = neighbors_dists.data();
    int64_t* neighbors_indices_ptr = neighbors_indices.data();
    if (mode == 0)
    {
        out =
            torch::from_blob(neighbors_indices_ptr, {query.size(0), max_count}, options = options);
        out_dists = torch::from_blob(neighbors_dists_ptr, {query.size(0), max_count},
                                     options = options_dist);
    }
    else if (mode == 1)
    {
        out = torch::from_blob(neighbors_indices_ptr, {(int)neighbors_indices.size() / 2, 2},
                               options = options);
        out_dists = torch::from_blob(neighbors_dists_ptr, {(int)neighbors_indices.size() / 2, 1},
                                     options = options_dist);
    }

    return std::make_pair(out.clone(), out_dists.clone());
}

at::Tensor degree(at::Tensor row, int64_t num_nodes)
{
    auto zero = at::zeros(num_nodes, row.options());
    auto one = at::ones(row.size(0), row.options());
    auto out = zero.scatter_add_(0, row, one);
    return out;
}

std::pair<at::Tensor, at::Tensor> batch_ball_query(at::Tensor support, at::Tensor query,
                                                   at::Tensor support_batch, at::Tensor query_batch,
                                                   float radius, int max_num, int mode, bool sorted)
{
    CHECK_CONTIGUOUS(support);
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(support_batch);
    CHECK_CONTIGUOUS(query_batch);

    at::Tensor idx;

    at::Tensor dist;
    std::vector<int64_t> neighbors_indices;
    std::vector<float> neighbors_dists;

    auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    auto options_dist = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    int max_count = 0;
    auto q_batch_access = query_batch.accessor<int64_t, 1>();
    auto s_batch_access = support_batch.accessor<int64_t, 1>();

    auto batch_size = q_batch_access[query_batch.size(0) - 1] + 1;
    TORCH_CHECK(batch_size == (s_batch_access[support_batch.size(0) - 1] + 1),
                "Both batches need to have the same number of samples.")

    query_batch = degree(query_batch, batch_size);
    query_batch = at::cat({at::zeros(1, query_batch.options()), query_batch.cumsum(0)}, 0);
    support_batch = degree(support_batch, batch_size);
    support_batch = at::cat({at::zeros(1, support_batch.options()), support_batch.cumsum(0)}, 0);
    std::vector<int64_t> query_batch_stl(query_batch.DATA_PTR<int64_t>(),
                                         query_batch.DATA_PTR<int64_t>() + query_batch.numel());
    std::vector<int64_t> support_batch_stl(support_batch.DATA_PTR<int64_t>(),
                                           support_batch.DATA_PTR<int64_t>() +
                                               support_batch.numel());

    AT_DISPATCH_ALL_TYPES(query.scalar_type(), "batch_radius_search", [&] {
        std::vector<scalar_t> queries_stl(query.DATA_PTR<scalar_t>(),
                                          query.DATA_PTR<scalar_t>() + query.numel());
        std::vector<scalar_t> supports_stl(support.DATA_PTR<scalar_t>(),
                                           support.DATA_PTR<scalar_t>() + support.numel());

        max_count = batch_nanoflann_neighbors<scalar_t>(
            queries_stl, supports_stl, query_batch_stl, support_batch_stl, neighbors_indices,
            neighbors_dists, radius, max_num, mode, sorted);
    });
    auto neighbors_dists_ptr = neighbors_dists.data();
    int64_t* neighbors_indices_ptr = neighbors_indices.data();

    if (mode == 0)
    {
        idx =
            torch::from_blob(neighbors_indices_ptr, {query.size(0), max_count}, options = options);
        dist = torch::from_blob(neighbors_dists_ptr, {query.size(0), max_count},
                                options = options_dist);
    }
    else if (mode == 1)
    {
        idx = torch::from_blob(neighbors_indices_ptr, {(int)neighbors_indices.size() / 2, 2},
                               options = options);
        dist = torch::from_blob(neighbors_dists_ptr, {(int)neighbors_indices.size() / 2, 1},
                                options = options_dist);
    }
    return std::make_pair(idx.clone(), dist.clone());
}

std::pair<at::Tensor, at::Tensor> dense_ball_query(at::Tensor support, at::Tensor query,
                                                   float radius, int max_num, int mode, bool sorted)
{
    CHECK_CONTIGUOUS(support);
    CHECK_CONTIGUOUS(query);

    int b = query.size(0);
    vector<at::Tensor> batch_idx;
    vector<at::Tensor> batch_dist;
    for (int i = 0; i < b; i++)
    {
        auto out_pair = ball_query(query[i], support[i], radius, max_num, mode, sorted);
        batch_idx.push_back(out_pair.first);
        batch_dist.push_back(out_pair.second);
    }
    auto out_idx = torch::stack(batch_idx);
    auto out_dist = torch::stack(batch_dist);
    return std::make_pair(out_idx, out_dist);
}
