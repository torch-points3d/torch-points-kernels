

#include "cloud.h"
#include "nanoflann.hpp"
#include <cstdint>
#include <set>


template <typename scalar_t>
int nanoflann_neighbors(std::vector<scalar_t>& queries, std::vector<scalar_t>& supports,
                        std::vector<int64_t>& neighbors_indices, std::vector<float>& dists, float radius,
                        int max_num, int mode, bool sorted);

template <typename scalar_t>
int batch_nanoflann_neighbors(std::vector<scalar_t>& queries, std::vector<scalar_t>& supports,
                              std::vector<int64_t>& q_batches, std::vector<int64_t>& s_batches,
                              std::vector<int64_t>& neighbors_indices, std::vector<float>& dists,
                              float radius, int max_num, int mode, bool sorted);

template <typename scalar_t>
void nanoflann_knn_neighbors(std::vector<scalar_t>& queries, std::vector<scalar_t>& supports,
                             std::vector<int64_t>& neighbors_indices, std::vector<float>& dists, int k);
