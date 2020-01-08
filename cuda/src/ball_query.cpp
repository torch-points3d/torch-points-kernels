#include "ball_query.h"
#include "utils.h"

void query_ball_point_kernel_dense_wrapper(int b, int n, int m, float radius,
					   int nsample, const float *new_xyz,
					   const float *xyz, int *idx);

void query_ball_point_kernel_partial_wrapper(int batch_size,
					     int size_new_xyz,
					     int size_xyz, float radius, int nsample,
					     const float *new_xyz,
					     const float *xyz,
					     long *idx_out,
					     long *dist_out);

at::Tensor ball_query_dense(at::Tensor new_xyz, at::Tensor xyz, const float radius,
			    const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_ball_point_kernel_dense_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
					  radius, nsample, new_xyz.data<float>(),
					  xyz.data<float>(), idx.data<int>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return idx;
}

std::pair<at::Tensor> ball_query_partial_dense(at::Tensor new_xyz,
					       at::Tensor xyz,
					       at::Tensor batch_new_xyz,
					       at::Tensor batch_xyz,
					       const float radius,
					       const int nsample) {
	CHECK_CONTIGUOUS(new_xyz);
	CHECK_CONTIGUOUS(xyz);
	CHECK_IS_FLOAT(new_xyz);
	CHECK_IS_FLOAT(xyz);
	CHECK_IS_LONG(batch_xyz);
	CHECK_IS_LONG(batch_new_xyz);
	if (new_xyz.type().is_cuda()) {
		CHECK_CUDA(xyz);
		CHECK_CUDA(batch_new_xyz);
		CHECK_CUDA(batch_xyz);
	}
	at::Tensor idx = torch::zeros({new_xyz.size(0), nsample},
				      at::device(new_xyz.device()).dtype(at::ScalarType::Long));
	at::Tensor dist = torch::zeros({new_xyz.size(0), nsample},
				      at::device(new_xyz.device()).dtype(at::ScalarType::Float));
	if (new_xyz.type().is_cuda()) {
		query_ball_point_kernel_partial_wrapper(batch_new_xyz[-1].item<long>(),
							new_xyz.size(0), xyz.size(0),
							radius, nsample,
							new_xyz.data<float>(),
							xyz.data<float>(),
							batch_new_xyz.data<long>(),
							batch_xyz.data<long>(),
							idx.data<long>()
							dist.data<float>());
	} else {
	  AT_CHECK(false, "CPU not supported");
	}

	return std::make_pair(idx, dist);
}
