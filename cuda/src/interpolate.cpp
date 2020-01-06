#include "interpolate.h"
#include "utils.h"

#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

extern THCState *state;

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points);

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.type().is_cuda()) {
    CHECK_CUDA(knows);
  }

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  if (unknowns.type().is_cuda()) {
    three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                            unknowns.data<float>(), knows.data<float>(),
                            dist2.data<float>(), idx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    three_interpolate_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    three_interpolate_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}

void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data<float>();
    const float *known = known_tensor.data<float>();
    float *dist2 = dist2_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx, stream);
}


void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                         at::Tensor points_tensor,
                         at::Tensor idx_tensor,
                         at::Tensor weight_tensor,
                         at::Tensor out_tensor) {

    const float *points = points_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *out = out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out, stream);
}

void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                            at::Tensor grad_out_tensor,
                            at::Tensor idx_tensor,
                            at::Tensor weight_tensor,
                            at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *grad_points = grad_points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points, stream);
}