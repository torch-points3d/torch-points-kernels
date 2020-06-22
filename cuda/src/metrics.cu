#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

void instance_iou_kernel_wrapper(int b, int n, int m, const float* dataset, float* temp,
                                            int* idxs);
