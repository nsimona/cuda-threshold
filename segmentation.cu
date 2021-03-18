#include "segmentation.h"
#define THREADS 16
#define thresholdVar 2

__global__ void histogramSum(int* g_idata, int* g_odata) {
    extern __shared__  int temp[];
    int tid = threadIdx.x;
    temp[tid] = g_idata[tid + blockIdx.x * blockDim.x];

    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d)  temp[tid] += temp[tid + d];
    }
    if (tid == 0) {
        //printf("temp[0] %d\n", temp[0]);
        g_odata[blockIdx.x] = temp[0];
    }
}

void reduction_gold(int* odata, int* idata, int len)
{
    *odata = 0;
    for (int i = 0; i < len; i++) *odata += idata[i];
}


__global__ void greyscale(unsigned char* inImg, int* outImg, int width, int height, int channels) {

    // IMPLEMENTS ALGORITHM FOR 3 CHANNEL GREYSCALE IMAGE
    //int x = threadIdx.x + blockIdx.x * blockDim.x;
    //int y = threadIdx.y + blockIdx.y * blockDim.y;

    //if (x < width && y < height) {
    //    int grayOffset = y * width + x;
    //    int rgbOffset = grayOffset * channels;
    //    unsigned char r = originalImg[rgbOffset];
    //    unsigned char g = originalImg[rgbOffset + 1];
    //    unsigned char b = originalImg[rgbOffset + 2];
    //    int offset = (r + g + b) / channels;
    //    for (int i = 0; i < channels; i++) {
    //        greyImg[rgbOffset + i] = offset;
    //    }
    //}

    // IMPLEMENTS ALGORITHM FOR 1 CHANNEL GREYSCALE IMAGE
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        int grayOffset = y * width + x;
        int rgbOffset = grayOffset * channels;
        unsigned char r = inImg[rgbOffset];
        unsigned char g = inImg[rgbOffset + 1];
        unsigned char b = inImg[rgbOffset + 2];
        outImg[grayOffset] = (int)(r + g + b) / 3;
        //printf("gray offset %d \n", outImg[grayOffset]);
    }
}

__device__ void calculateThresholdValues(int mean, int* thresholdValue) {
    int step = mean / thresholdVar;
    int currentValue = 0;
    for (int i = 0; i < thresholdVar; i++) {
        currentValue += step;
        *(thresholdValue + i) = currentValue;
        //thresholdValue[i] = currentValue;
    }
}

__device__ void calculateGreyValues(int* greyValue) {
    int maxValue = 255;
    int step = maxValue / thresholdVar;
    int currentValue = 0;
    for (int i = 0; i < thresholdVar; i++) {
        currentValue += step;
        *(greyValue + i) = currentValue;
    }
}

__global__ void threshold(int* inImg, unsigned char* outImg, unsigned int width, unsigned int height, int channels, int mean, int variable = 0) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pixel = y * width + x;
    int value = 0;
    int thresholdValues[thresholdVar];
    int greyValues[thresholdVar];

    int* thresholdVariable = &thresholdValues[0];
    int* greys = &greyValues[0];

    if (variable) {
        calculateThresholdValues(mean, thresholdVariable);
        calculateGreyValues(greyValues);
    }


    if (x < width && y < height) {
        if (inImg[pixel] > mean) value = 255;
        if (variable) {
            if (inImg[pixel] < mean && inImg[pixel] > * (thresholdVariable + 1)) value = *(greys + variable);
            if (inImg[pixel] < *(thresholdVariable + 1) && inImg[pixel] > * (thresholdVariable + 0)) value = *(greys + variable - 1);
        }
        outImg[pixel * channels] = value;
        outImg[pixel * channels + 1] = value;
        outImg[pixel * channels + 2] = value;
    }

}
void convertToGreyscale(unsigned char* inImg, int* outImg, int width, int height, int channels)
{
    dim3 dimGrid = dim3((width / THREADS) + 1, (height / THREADS) + 1, 1);
    dim3 dimBlock = dim3(THREADS, THREADS, 1);
    unsigned char* d_originalImg = NULL;
    int* d_greyImg = NULL;
    int size = width * height;

    cudaMalloc((void**)&d_originalImg, size * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_greyImg, size * sizeof(int));

    cudaMemcpy(d_originalImg, inImg, size * channels, cudaMemcpyHostToDevice);

    greyscale << <dimGrid, dimBlock >> > (d_originalImg, d_greyImg, width, height, channels);

    cudaMemcpy(outImg, d_greyImg, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_originalImg);
    cudaFree(d_greyImg);
}

void thresholdFilter(int* inImg, unsigned char* outImg, int width, int height, int channels, int variableThreshold = 0)
{
    dim3 dimGrid = dim3((width / THREADS) + 1, (height / THREADS) + 1, 1);
    dim3 dimBlock = dim3(THREADS, THREADS, 1);
    int size = width * height;

    // CALCULATES HISTOGRAM MEAN VALUE 

    int* host_sum = (int*)malloc(sizeof(int) * size);
    int* d_idata;
    int* d_odata;
    int sharedMemSize = sizeof(int) * THREADS;
    int sumResult = 0;
    int mean;

    // Using histogram sum on host
    int host_calculated;
    reduction_gold(&host_calculated, inImg, size);

    //int* dev_lastBlockCounter;
    //cudaMalloc((void**)&dev_lastBlockCounter, sizeof(int));
    //cudaMemset(dev_lastBlockCounter, 0, sizeof(int));

    cudaMalloc((void**)&d_idata, size * sizeof(int));
    cudaMalloc((void**)&d_odata, size * sizeof(int));

    cudaMemcpy(d_idata, inImg, size * sizeof(int), cudaMemcpyHostToDevice);
    ////sumCommMultiBlock << <gridSize, blockSize >> > (d_idata, size, d_odata, dev_lastBlockCounter);
    histogramSum << < 1, size, sharedMemSize >> > (d_idata, d_odata);

    cudaMemcpy(host_sum, d_odata, size * sizeof(int), cudaMemcpyDeviceToHost);
    //reduction_gold(&sumResult, host_sum, size/THREADS);
    ////cudaMemcpy(&d_mean, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_odata);
    cudaFree(d_idata);

    int using_cuda_mean = (int)host_sum / size;
    mean = (int)host_calculated / size;

    //printf(" calucalted by device host_sum %d size % d mean %d\n", host_sum[0], size, using_cuda_mean);
    //printf(" calucalted by host \host_calculated %d size % d mean %d\n", host_calculated, size, mean);

    int* d_grey = NULL;
    unsigned char* d_threshold = NULL;


    cudaMalloc((void**)&d_grey, size * sizeof(int));
    cudaMalloc((void**)&d_threshold, size * channels);

    cudaMemcpy(d_grey, inImg, size * sizeof(int), cudaMemcpyHostToDevice);
    if (variableThreshold) {
        threshold << <dimGrid, dimBlock >> > (d_grey, d_threshold, width, height, channels, mean, 1);
    }
    else {
        threshold << <dimGrid, dimBlock >> > (d_grey, d_threshold, width, height, channels, mean);
    }

    cudaMemcpy(outImg, d_threshold, size * channels, cudaMemcpyDeviceToHost);

    cudaFree(d_grey);
    cudaFree(d_threshold);
}
