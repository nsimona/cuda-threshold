#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


void reduction_gold(int* odata, int* idata, int len);
void convertToGreyscale(unsigned char* inImg, int* outImg, int width, int height, int channels);
void thresholdFilter(int* inImg, unsigned char* outImg, int width, int height, int channels, int variableThreshold);
