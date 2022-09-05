#!/bin/bash

export PYTHON_VERSION=$1
export TORCH_VERSION=$2
export CUDA_VERSION=$3


# CUDA_HOME needs to be set to build this package but sometimes it is incorrectly set to an incompatible version when the build environment is created.
# This is the case if the system has multiple CUDA installations and/or an incompatible version (not 10.1, 10.2, 11.1) is the default. Rolling release Linux distributions do this.
# A solution is to accept a optional path to the correct CUDA installation.
if [ -z "$4" ]; then
  export CUDA_HOME=$4
  
  # Also need to set gcc and g++ versions to the matching CUDA version. See the table in https://stackoverflow.com/a/46380601/8724072
  export CC=$5
  export CXX=$6

else
  echo "Using default CUDA_HOME="$CUDA_HOME

fi



export CONDA_PYTORCH_CONSTRAINT="pytorch==${TORCH_VERSION%.*}.*"

if [ "${CUDA_VERSION}" = "cpu" ]; then
  export CONDA_CUDATOOLKIT_CONSTRAINT="cpuonly  # [not osx]"
else
  case $CUDA_VERSION in
    cu111)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.1.*"
      ;;
    cu102)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.2.*"
      ;;
    cu101)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.1.*"
      ;;
    *)
      echo "Unrecognized CUDA_VERSION=$CUDA_VERSION"
      exit 1
      ;;
  esac
fi

echo "PyTorch $TORCH_VERSION+$CUDA_VERSION"
echo "- $CONDA_PYTORCH_CONSTRAINT"
echo "- $CONDA_CUDATOOLKIT_CONSTRAINT"

conda build . -c pytorch -c defaults -c conda-forge -c rusty1s --output-folder "$HOME/conda-bld"
