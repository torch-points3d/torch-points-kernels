#include <torch/extension.h>

#include "compat.h"
#include "utils.h"

at::Tensor get_dist(at::Tensor x, ptrdiff_t index)
{
    return (x - x[index]).norm(2, 1);
}

at::Tensor fps(at::Tensor points, const int nsamples, bool random)
{
    CHECK_CONTIGUOUS(points);

    auto out_options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    auto batch_size = points.size(0);
    auto out = torch::empty({batch_size, nsamples}, out_options);
    auto out_a = out.accessor<int64_t, 2>();

    for (ptrdiff_t b = 0; b < batch_size; b++)
    {
        auto y = points[b];
        ptrdiff_t start = 0;
        if (random)
            start = at::randperm(y.size(0), out_options).DATA_PTR<int64_t>()[0];

        out_a[b][0] = start;
        auto dist = get_dist(y, start);
        for (ptrdiff_t i = 1; i < nsamples; i++)
        {
            ptrdiff_t argmax = dist.argmax().DATA_PTR<int64_t>()[0];
            out_a[b][i] = argmax;
            dist = at::min(dist, get_dist(y, argmax));
        }
    }
    return out;
}