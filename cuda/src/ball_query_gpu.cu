#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

#define THREADS 1024

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel_dense(int b, int n, int m, float radius,
					      int nsample,
					      const float *__restrict__ new_xyz,
					      const float *__restrict__ xyz,
					      int *__restrict__ idx_out) {

  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx_out += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx_out[j * nsample + l] = k;
          }
        }
        idx_out[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

__global__ void query_ball_point_kernel_partial_dense(int size_x,
						      int size_y,
						      float radius,
						      int nsample,
						      const float *__restrict__ x,
						      const float *__restrict__ y,
						      const long *__restrict__ batch_x,
						      const long *__restrict__ batch_y,
						      int64_t *__restrict__ idx_out,
							  float * __restrict__ dist_out) {

	// taken from https://github.com/rusty1s/pytorch_cluster/blob/master/cuda/radius_kernel.cu
	const ptrdiff_t batch_idx = blockIdx.x;
	const ptrdiff_t idx = threadIdx.x;

	const ptrdiff_t start_idx_q = batch_x[batch_idx];
	const ptrdiff_t end_idx_q = batch_x[batch_idx + 1];

	const ptrdiff_t start_idx_s = batch_y[batch_idx];
	const ptrdiff_t end_idx_s = batch_y[batch_idx + 1];
	float radius2 = radius * radius;

	for (ptrdiff_t n_q = start_idx_q + idx; n_q < end_idx_q; n_q += THREADS) {
		size_t count = 0;
		for (ptrdiff_t n_s = start_idx_s; n_s < end_idx_s; n_s++) {
			float dist = 0;
			for (ptrdiff_t d = 0; d < 3; d++) {
				dist += (x[n_q * 3 + d] - y[n_s * 3 + d]) *
					(x[n_q * 3 + d] - y[n_s * 3 + d]);
			}
			if(dist <= radius2){
				if (count == 0){
					for(ptrdiff_t l = 0; l < nsample; ++l){
						idx_out[n_s * nsample + l] = n_q;
						dist_out[n_s * nsample + l] = dist;
					}
				}
				idx_out[n_s * nsample + count] = n_q;
				dist_out[n_s * nsample + count] = dist;
				count++;
		       }
			if(count >= nsample){
				break;
			}
		}
	}
}

void query_ball_point_kernel_dense_wrapper(int b, int n, int m, float radius,
					   int nsample, const float *new_xyz,
					   const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel_dense<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}

void query_ball_point_kernel_partial_wrapper(long batch_size,
						int size_x,
						int size_y, 
						float radius, 
						int nsample,
						const float *x,
						const float *y,
						const long *batch_x,
						const long *batch_y,
						int64_t *idx_out,
						float *dist_out) {

	query_ball_point_kernel_partial_dense<<<batch_size, THREADS>>>(
		size_x, size_y, radius, nsample, x, y,
		batch_x, batch_y, idx_out, dist_out);

	CUDA_CHECK_ERRORS();
}
