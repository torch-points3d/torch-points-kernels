
// Taken from https://github.com/HuguesTHOMAS/KPConv

#include "neighbors.h"
#include <chrono>
#include <random>

template <typename scalar_t>
int nanoflann_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
                        vector<int64_t>& neighbors_indices, vector<float>& dists, float radius,
                        int max_num, int mode, bool sorted)
{
    // Initiate variables
    // ******************
    std::random_device rd;
    std::mt19937 g(rd());

    // square radius
    const float search_radius = static_cast<float>(radius * radius);

    // indices
    int i0 = 0;

    // Counting vector
    size_t max_count = 1;

    // Nanoflann related variables
    // ***************************

    // CLoud variable
    PointCloud<scalar_t> pcd;
    pcd.set(supports);

    // Cloud query
    PointCloud<scalar_t> pcd_query;
    pcd_query.set(queries);

    // Tree parameters
    nanoflann::KDTreeSingleIndexAdaptorParams tree_params(15 /* max leaf */);

    // KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<scalar_t, PointCloud<scalar_t>>, PointCloud<scalar_t>, 3>
        my_kd_tree_t;

    // Pointer to trees
    std::unique_ptr<my_kd_tree_t> index(new my_kd_tree_t(3, pcd, tree_params));
    index->buildIndex();
    // Search neigbors indices
    // ***********************

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = sorted;
    auto num_query_points = pcd_query.get_point_count();
    std::vector<std::vector<std::pair<size_t, scalar_t>>> list_matches(num_query_points);

    for (size_t i = 0; i < num_query_points; i++)
    {
        // Find neighbors
        list_matches[i0].reserve(max_count);
        std::vector<std::pair<size_t, scalar_t>> ret_matches;

        const size_t nMatches = index->radiusSearch(pcd_query.get_point_ptr(i), search_radius,
                                                    ret_matches, search_params);
        if (nMatches == 0)
            list_matches[i0] = {std::make_pair(0, -1)};
        else
        {
            if (!sorted)
                std::shuffle(ret_matches.begin(), ret_matches.end(), g);
            list_matches[i0] = ret_matches;
        }
        max_count = max(max_count, nMatches);
        i0++;
    }
    // Reserve the memory
    if (max_num > 0)
    {
        max_count = max_num;
    }
    if (mode == 0)
    {
        neighbors_indices.resize(list_matches.size() * max_count, 0);
        dists.resize(list_matches.size() * max_count, -1);
        i0 = 0;
        int token = 0;
        for (auto& inds : list_matches)
        {
            token = inds[0].first;
            for (size_t j = 0; j < max_count; j++)
            {
                if (j < inds.size())
                {
                    neighbors_indices[i0 * max_count + j] = inds[j].first;
                    dists[i0 * max_count + j] = (float)inds[j].second;
                }
                else
                {
                    neighbors_indices[i0 * max_count + j] = token;
                    dists[i0 * max_count + j] = -1;
                }
            }
            i0++;
        }
    }
    else if (mode == 1)
    {
        size_t size = 0; // total number of edges
        for (auto& inds : list_matches)
        {
            if (inds.size() <= max_count)
                size += inds.size();
            else
                size += max_count;
        }
        neighbors_indices.resize(size * 2);
        dists.resize(size);
        int i0 = 0; // index of the query points
        int u = 0;  // curent index of the neighbors_indices
        for (auto& inds : list_matches)
        {
            for (size_t j = 0; j < max_count; j++)
            {
                if (j < inds.size())
                {
                    neighbors_indices[u] = inds[j].first;
                    neighbors_indices[u + 1] = i0;
                    dists[u / 2] = (float)inds[j].second;
                    u += 2;
                }
            }
            i0++;
        }
    }
    return max_count;
}

template <typename scalar_t>
int batch_nanoflann_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
                              vector<int64_t>& q_batches, vector<int64_t>& s_batches,
                              vector<int64_t>& neighbors_indices, vector<float>& dists,
                              float radius, int max_num, int mode, bool sorted)
{
    // Initiate variables
    // ******************
    std::random_device rd;
    std::mt19937 g(rd());

    // indices
    int i0 = 0;

    // Square radius
    float r2 = radius * radius;

    // Counting vector
    int max_count = 0;

    // batch index
    int b = 0;

    // Nanoflann related variables
    // ***************************

    // CLoud variable
    PointCloud<scalar_t> current_cloud;
    PointCloud<scalar_t> query_pcd;
    query_pcd.set(queries);
    auto num_query_points = query_pcd.get_point_count();
    vector<vector<pair<size_t, scalar_t>>> all_inds_dists(num_query_points);

    // Tree parameters
    nanoflann::KDTreeSingleIndexAdaptorParams tree_params(15 /* max leaf */);

    // KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<scalar_t, PointCloud<scalar_t>>, PointCloud<scalar_t>, 3>
        my_kd_tree_t;

    // Build KDTree for the first batch element
    current_cloud.set_batch(supports, s_batches[b], s_batches[b + 1]);
    std::unique_ptr<my_kd_tree_t> index(new my_kd_tree_t(3, current_cloud, tree_params));
    index->buildIndex();

    // Search neigbors indices
    // ***********************
    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = sorted;
    for (size_t i = 0; i < num_query_points; i++)
    {
        // Check if we changed batch
        if (i0 == q_batches[b + 1] && b < (int)s_batches.size() - 1 &&
            b < (int)q_batches.size() - 1)
        {
            // Change the points
            b++;
            if (s_batches[b] < s_batches[b + 1])
                current_cloud.set_batch(supports, s_batches[b], s_batches[b + 1]);

            index.reset(new my_kd_tree_t(3, current_cloud, tree_params));
            index->buildIndex();
        }

        // Find neighboors
        std::vector<std::pair<size_t, scalar_t>> ret_matches;
        ret_matches.reserve(max_count);
        size_t nMatches =
            index->radiusSearch(query_pcd.get_point_ptr(i), r2, ret_matches, search_params);

        // Shuffle if needed
        if (!sorted)
            std::shuffle(ret_matches.begin(), ret_matches.end(), g);
        all_inds_dists[i0] = ret_matches;

        // Update max count
        if (nMatches > (size_t)max_count)
            max_count = nMatches;
        // Increment query idx
        i0++;
    }
    // how many neighbors do we keep
    if (max_num > 0)
        max_count = max_num;

    const int token = -1;
    if (mode == 0)
    {
        neighbors_indices.resize(query_pcd.get_point_count() * max_count);
        dists.resize(query_pcd.get_point_count() * max_count);
        i0 = 0;
        b = 0;

        for (auto& inds_dists : all_inds_dists)
        { // Check if we changed batch
            if (i0 == q_batches[b + 1] && b < (int)s_batches.size() - 1 &&
                b < (int)q_batches.size() - 1)
                b++;

            for (int j = 0; j < max_count; j++)
            {
                if ((size_t)j < inds_dists.size())
                {
                    neighbors_indices[i0 * max_count + j] = inds_dists[j].first + s_batches[b];
                    dists[i0 * max_count + j] = (float)inds_dists[j].second;
                }
                else
                {
                    neighbors_indices[i0 * max_count + j] = token;
                    dists[i0 * max_count + j] = -1;
                }
            }
            i0++;
        }
        index.reset();
    }
    else if (mode == 1)
    {
        int size = 0; // total number of edges
        for (auto& inds_dists : all_inds_dists)
        {
            if ((int)inds_dists.size() <= max_count)
                size += inds_dists.size();
            else
                size += max_count;
        }
        neighbors_indices.resize(size * 2);
        dists.resize(size);
        i0 = 0;
        b = 0;
        int u = 0;
        for (auto& inds_dists : all_inds_dists)
        {
            if (i0 == q_batches[b + 1] && b < (int)s_batches.size() - 1 &&
                b < (int)q_batches.size() - 1)
            {
                b++;
            }
            for (int j = 0; j < max_count; j++)
            {
                if ((unsigned int)j < inds_dists.size())
                {
                    neighbors_indices[u] = inds_dists[j].first + s_batches[b];
                    neighbors_indices[u + 1] = i0;
                    dists[u / 2] = (float)inds_dists[j].second;
                    u += 2;
                }
            }
            i0++;
        }
    }
    return max_count;
}

template <typename scalar_t>
void nanoflann_knn_neighbors(vector<scalar_t>& queries, vector<scalar_t>& supports,
                             vector<int64_t>& neighbors_indices, vector<float>& dists, int k)
{
    // Nanoflann related variables
    // ***************************
    // CLoud variable
    PointCloud<scalar_t> pcd;
    pcd.set(supports);
    // Cloud query
    PointCloud<scalar_t> pcd_query;
    pcd_query.set(queries);

    // Tree parameters
    nanoflann::KDTreeSingleIndexAdaptorParams tree_params(15 /* max leaf */);

    // KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<scalar_t, PointCloud<scalar_t>>, PointCloud<scalar_t>, 3>
        my_kd_tree_t;

    // Pointer to trees
    std::unique_ptr<my_kd_tree_t> index(new my_kd_tree_t(3, pcd, tree_params));
    index->buildIndex();

    // Search neigbors indices
    // ***********************
    size_t current_pos = 0;
    auto num_query_points = pcd_query.get_point_count();
    for (size_t i = 0; i < num_query_points; i++)
    {
        // Find neighbors
        std::vector<size_t> ret_index(k);
        std::vector<scalar_t> out_dist_sqr(k);

        const size_t nMatches =
            index->knnSearch(pcd_query.get_point_ptr(i), k, &ret_index[0], &out_dist_sqr[0]);
        for (size_t i = 0; i < nMatches; i++)
        {
            neighbors_indices[i + current_pos] = ret_index[i];
            dists[i + current_pos] = out_dist_sqr[i];
        }
        current_pos += k;
    }
}