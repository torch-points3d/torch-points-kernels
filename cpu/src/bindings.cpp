#include "group_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_points", &group_points);
}