from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
    CppExtension,
)
import glob

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
extra_compile_args = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ["-DVERSION_GE_1_3"]

ext_src_root = "cuda"
ext_sources = glob.glob("{}/src/*.cpp".format(ext_src_root)) + glob.glob("{}/src/*.cu".format(ext_src_root))

ext_modules = []
if CUDA_HOME:
    ext_modules.append(
        CUDAExtension(
            name="torch_points_kernels.points_cuda",
            sources=ext_sources,
            include_dirs=["{}/include".format(ext_src_root)],
            extra_compile_args={"cxx": extra_compile_args, "nvcc": extra_compile_args,},
        )
    )

cpu_ext_src_root = "cpu"
cpu_ext_sources = glob.glob("{}/src/*.cpp".format(cpu_ext_src_root))

ext_modules.append(
    CppExtension(
        name="torch_points_kernels.points_cpu",
        sources=cpu_ext_sources,
        include_dirs=["{}/include".format(cpu_ext_src_root)],
        extra_compile_args={"cxx": extra_compile_args,},
    )
)

requirements = ["torch>=1.1.0"]

url = 'https://github.com/nicolas-chaulet/torch-points-kernels'
__version__="0.5.2"
setup(
    name="torch-points-kernels",
    version=__version__,
    author="Nicolas Chaulet",
    packages=find_packages(),
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    install_requires=requirements,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    long_description=long_description,
    long_description_content_type='text/markdown'
)
