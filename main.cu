#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "segmentation.h"

int main()
{
    // read the image
    int width, height, channels = 0;
    unsigned char* originalImg = stbi_load("sample/apple.jpg", &width, &height, &channels, 0);

    int img_size = width * height * channels;
    int img_grey_size = width * height;
    int* greyImg = (int*)malloc(img_grey_size * sizeof(int));
    unsigned char* thresholdImg = (unsigned char*)malloc(img_size);
    unsigned char* variableThresholdImg = (unsigned char*)malloc(img_size);

    // convert to 1 channel array
    convertToGreyscale(originalImg, greyImg, width, height, channels);
    // threshold filter 
    thresholdFilter(greyImg, thresholdImg, width, height, channels, 0);
    stbi_write_jpg("sample/apple_threshold.jpg", width, height, channels, thresholdImg, 100);
    // threshold variable filter 
    thresholdFilter(greyImg, variableThresholdImg, width, height, channels, 1);
    stbi_write_jpg("sample/apple_threshold-variable.jpg", width, height, channels, variableThresholdImg, 100);

    // TODO kmeans filter
    //unsigned char* kmeansImg = (unsigned char*)malloc(img_size);
    //kmeansFilter(originalImg, kmeansImg, width, height, channels);
    //stbi_write_jpg("kmeans.jpg", width, height, channels, kmeansImg, 100);

    return 0;

}
