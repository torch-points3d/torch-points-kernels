#include "ball_query.h"
#include "interpolate.h"
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("furthest_point_sampling", &furthest_point_sampling);

    m.def("three_nn", &three_nn);
    m.def("three_interpolate", &three_interpolate);
    m.def("three_interpolate_grad", &three_interpolate_grad);

    m.def("ball_query_dense", &ball_query_dense);
    m.def("ball_query_partial_dense", &ball_query_partial_dense);
}
