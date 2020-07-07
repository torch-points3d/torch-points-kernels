# 3D Point Cloud Kernels
Pytorch CPU and CUDA kernels for spatial search and interpolation for 3D point clouds.

[![PyPI version](https://badge.fury.io/py/torch-points-kernels.svg)](https://badge.fury.io/py/torch-points-kernels) ![Deploy](https://github.com/nicolas-chaulet/torch-points-kernels/workflows/Deploy/badge.svg) ![Unittests](https://github.com/nicolas-chaulet/torch-points-kernels/workflows/Unittests/badge.svg)

## Installation
**Requires torch** version 1.0 or higher to be installed before proceeding. Once this is done, simply run
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

## Troubleshooting

### Compilation issues
Ensure that at least PyTorch 1.4.0 is installed and verify that `cuda/bin` and `cuda/include` are in your `$PATH` and `$CPATH` respectively, e.g.:
```
$ python -c "import torch; print(torch.__version__)"
>>> 1.4.0

$ python -c "import torch; print(torch.__version__)"
>>> 1.1.0

$ echo $PATH
>>> /usr/local/cuda/bin:...

$ echo $CPATH
>>> /usr/local/cuda/include:...
```


### CUDA kernel failed : no kernel image is available for execution on the device

This can happen when trying to run the code on a different GPU than the one used to compile the `torch-points-kernels` library. Uninstall `torch-points-kernels`, clear cache, and reinstall after setting the `TORCH_CUDA_ARCH_LIST` environment variable. For example, for compiling with a Tesla T4 (Turing 7.5) and running the code on a Tesla V100 (Volta 7.0) use:
```
export TORCH_CUDA_ARCH_LIST="7.0;7.5"
```
See [this useful chart](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) for more architecture compatibility.


## Projects using those kernels.

[```Pytorch Point Cloud Benchmark```](https://github.com/nicolas-chaulet/deeppointcloud-benchmarks)

## Credit

* [```Pointnet2_Tensorflow```](https://github.com/charlesq34/pointnet2) by [Charles R. Qi](https://github.com/charlesq34)

* [```Pointnet2_PyTorch```](https://github.com/erikwijmans/Pointnet2_PyTorch) by [Erik Wijmans](https://github.com/erikwijmans)

* [```GRNet```](https://github.com/hzxie/GRNet) by [Haozhe Xie](https://github.com/hzxie)
