from setuptools import setup, find_packages
import os
import glob

try:
    import torch
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        CUDA_HOME,
        CppExtension,
    )
except:
    raise ModuleNotFoundError("Please install pytorch >= 1.1 before proceeding.")

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
WITH_CPU = True
if os.getenv('FORCE_CUDA', '0') == '1':
    WITH_CUDA = True
if os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    WITH_CUDA = True
    WITH_CPU = False
if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    WITH_CUDA = False
    WITH_CPU = True

def get_ext_modules():
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    extra_compile_args = ["-O3"]
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
        extra_compile_args += ["-DVERSION_GE_1_3"]

    ext_src_root = "cuda"
    ext_sources = glob.glob("{}/src/*.cpp".format(ext_src_root)) + glob.glob("{}/src/*.cu".format(ext_src_root))

    ext_modules = []
    if WITH_CUDA:
        ext_modules.append(
            CUDAExtension(
                name="torch_points_kernels.points_cuda",
                sources=ext_sources,
                include_dirs=["{}/include".format(ext_src_root)],
                extra_compile_args={
                    "cxx": extra_compile_args,
                    "nvcc": extra_compile_args,
                },
            )
        )

    cpu_ext_src_root = "cpu"
    cpu_ext_sources = glob.glob("{}/src/*.cpp".format(cpu_ext_src_root))

    if WITH_CPU:
        ext_modules.append(
            CppExtension(
                name="torch_points_kernels.points_cpu",
                sources=cpu_ext_sources,
                include_dirs=["{}/include".format(cpu_ext_src_root)],
                extra_compile_args={
                    "cxx": extra_compile_args,
                },
            )
        )
    return ext_modules


class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, no_python_abi_suffix=True, use_ninja=False, **kwargs)


def get_cmdclass():
    return {"build_ext": CustomBuildExtension}

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = ["torch>=1.1.0", "numba", "numpy<1.20", "scikit-learn"]

url = "https://github.com/nicolas-chaulet/torch-points-kernels"
__version__ = "0.7.0"
setup(
    name="torch-points-kernels",
    version=__version__,
    author="Nicolas Chaulet",
    packages=find_packages(),
    description="PyTorch kernels for spatial operations on point clouds",
    url=url,
    download_url="{}/archive/{}.tar.gz".format(url, __version__),
    install_requires=requirements,
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
