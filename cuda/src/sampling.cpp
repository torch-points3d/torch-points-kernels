#include "sampling.h"
#include "compat.h"
#include "utils.h"

void furthest_point_sampling_kernel_wrapper(int b, int n, int m, const float* dataset, float* temp,
                                            int* idxs);

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples)
{
    CHECK_CONTIGUOUS(points);
    CHECK_IS_FLOAT(points);
    CHECK_CUDA(points);

    at::Tensor output = torch::zeros({points.size(0), nsamples},
                                     at::device(points.device()).dtype(at::ScalarType::Int));

    at::Tensor tmp = torch::full({points.size(0), points.size(1)}, 1e10,
                                 at::device(points.device()).dtype(at::ScalarType::Float));

    furthest_point_sampling_kernel_wrapper(points.size(0), points.size(1), nsamples,
                                           points.DATA_PTR<float>(), tmp.DATA_PTR<float>(),
                                           output.DATA_PTR<int>());

    return output;
}
