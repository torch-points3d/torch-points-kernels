## Usage
```
./build_conda.sh python, pytorch, cuda version, [cuda_home path, gcc path, g++ path]
```
## Example
```
./build_conda.sh 3.9 1.9.0 cu111 #assume CUDA installation by current environment variables
./build_conda.sh 3.9 1.9.0 cu111 /usr/local/cuda /usr/bin/gcc /usr/bin/g++ #Ubuntu
./build_conda.sh 3.8 1.10.2 cu102 /opt/cuda-10.2/ /opt/cuda-10.2/bin/gcc /opt/cuda-10.2/bin/g++ #Arch Linux uses /opt/
```

## Notes
Your gcc and g++ versions must match the correct version of CUDA you are trying to build for. See [this table](https://stackoverflow.com/a/46380601/8724072) for valid configurations. To check your CUDA, GCC and g++ versions, run the following commands respectively. If they do not match you will need to install another version and specify the paths manually as shown above.
```
nvcc --version
gcc --version
g++ --version
```