#include "interpolate.h"
#include "compat.h"
#include "utils.h"

std::vector<torch::Tensor> three_nn(torch::Tensor unknowns, torch::Tensor knows)
{
    CHECK_CONTIGUOUS(unknowns);
    CHECK_CONTIGUOUS(knows);
    CHECK_IS_FLOAT(unknowns);
    CHECK_IS_FLOAT(knows);

    CHECK_CUDA(knows);
    CHECK_CUDA(unknowns);

    // at::Tensor idx = torch::zeros({unknowns.size(0), unknowns.size(1), 3},
    //                               at::device(unknowns.device()).dtype(at::ScalarType::Int));
    // at::Tensor dist2 = torch::zeros({unknowns.size(0), unknowns.size(1), 3},
    //                                 at::device(unknowns.device()).dtype(at::ScalarType::Float));

    // three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
    //                         unknowns.DATA_PTR<float>(), knows.DATA_PTR<float>(),
    //                         dist2.DATA_PTR<float>(), idx.DATA_PTR<int>());

    return three_nn_kernel_wrapper(unknowns, knows);

    // return {dist2, idx};
}

torch::Tensor three_interpolate(torch::Tensor points, torch::Tensor idx, torch::Tensor weight)
{
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    CHECK_CUDA(idx);
    CHECK_CUDA(weight);

    return three_interpolate_kernel_wrapper(points, idx, weight);

    // return output;
}
torch::Tensor three_interpolate_grad(torch::Tensor grad_out, torch::Tensor idx,
                                     torch::Tensor weight, const int m)
{
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
    CHECK_CUDA(grad_out);

    return three_interpolate_grad_kernel_wrapper(grad_out, idx, weight, m);

    // return output;
}
