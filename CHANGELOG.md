# 0.6.5

## Additions
- Clustering algorithm for [PointGroup](https://arxiv.org/pdf/2004.01658.pdf)
- Instance IoU computation on CPU and GPU

## Change
- Force no ninja for the compilation

# 0.6.4

## Bug fix
- CPU version works for MacOS

# 0.6.2

## Bug fix
- Fix install with pip > 19

# 0.6.1

## Bug fix
- Random memory access on cpu radius search in the degree function

# 0.6.0

## Bug fix
- Require pytorch implicitely and log nice message when missing

# 0.5.3

## Update
- ball query returns squared distance instead of distance
- leaner Point Cloud struct that avoids copying data

## Bug fix
- Package would not install if pytorch is not already installed
