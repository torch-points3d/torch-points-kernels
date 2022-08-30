#include "sampling.h"
#include "compat.h"
#include "utils.h"

torch::Tensor furthest_point_sampling_kernel_wrapper(torch::Tensor points, const int nsamples);

torch::Tensor furthest_point_sampling(torch::Tensor points, const int nsamples)
{
    CHECK_CONTIGUOUS(points);
    CHECK_IS_FLOAT(points);
    CHECK_CUDA(points);

    return furthest_point_sampling_kernel_wrapper(points, nsamples);
}
