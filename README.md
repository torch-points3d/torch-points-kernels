# 3D Point Cloud Kernels
Pytorch CPU and CUDA kernels for spatial search and interpolation for 3D point clouds.

![Deploy](https://github.com/nicolas-chaulet/torch-points-kernels/workflows/Deploy/badge.svg) ![Unittests](https://github.com/nicolas-chaulet/torch-points-kernels/workflows/Unittests/badge.svg)

## Installation
Requires torch version 1.0 or higher
```
pip install torch-points-kernels
```
or with poetry:
```
poetry add torch-points-kernels
```

## Usage
```
import torch
import torch_points_kernels.points_cuda
```

## Build and test
```
python setup.py build_ext --inplace
python -m unittest
```

## Projects using those kernels.

[```Pytorch Point Cloud Benchmark```](https://github.com/nicolas-chaulet/deeppointcloud-benchmarks)

## Credit

* [```Pointnet2_Tensorflow```](https://github.com/charlesq34/pointnet2) by [Charles R. Qi](https://github.com/charlesq34)

* [```Pointnet2_PyTorch```](https://github.com/erikwijmans/Pointnet2_PyTorch) by [Erik Wijmans](https://github.com/erikwijmans)
