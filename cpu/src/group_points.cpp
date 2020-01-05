#include "group_points.h"
#include "utils.h"

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
at::Tensor group_points(at::Tensor points, at::Tensor idx) {
    CHECK_CPU(points);
    CHECK_CPU(idx);

    at::Tensor output = torch::zeros(
        {points.size(0), points.size(1), idx.size(1), idx.size(2)}, 
        at::device(points.device()).dtype(at::ScalarType::Float)
    ); 

    for (int batch_index = 0; batch_index < output.size(0); batch_index++) {
        for (int feat_index = 0; feat_index < output.size(1); feat_index++) {
            for (int point_index = 0; point_index < output.size(2); point_index++) {
                for (int sample_index = 0; sample_index < output.size(3); sample_index++) {
                    output[batch_index][feat_index][point_index][sample_index] 
                    = points[batch_index][feat_index][
                        idx[batch_index][point_index][sample_index]
                    ]; 
                }
            }
        }
    }

    return output; 
}
