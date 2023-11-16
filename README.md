# CUDA Image Fragmentation and Segmentation

## Introduction

This project focuses on the implementation of two image processing algorithms, namely Variable Threshold and Multiple Threshold, utilizing CUDA (Compute Unified Device Architecture) for parallel computation. Image segmentation is a critical step in computer vision and graphics, aiding in the extraction of meaningful information from images by dividing them into regions of interest.

### Variable Threshold Algorithm

The Variable Threshold algorithm involves determining optimal thresholds for image segmentation based on pixel intensities. It dynamically adjusts the threshold values, enhancing adaptability to varying image characteristics.

## Implementation Details

The project is implemented in the C programming language, leveraging the power of CUDA for parallel processing on NVIDIA GPUs.

## Prerequisites

Ensure you have the following prerequisites installed before using this project:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): The project leverages CUDA for parallel computing, so make sure you have the CUDA Toolkit installed on your system.

- [Visual Studio](https://visualstudio.microsoft.com/): The project is developed using Visual Studio. You can use the Community edition, which is free and suitable for CUDA development.

- [NVIDIA GPU](https://www.nvidia.com/): A CUDA-compatible NVIDIA GPU is required to execute the parallelized code efficiently.

## Installation

1. Clone the Repository

```bash
   git https://github.com/nsimona/cuda-threshold.git
   cd cuda-threshold
```

2. Open in Visual Studio

Open the project solution file (.sln) in Visual Studio.

3. Build and Run

Build the solution in Visual Studio, ensuring that the CUDA-enabled GPU is selected. Run the project to execute the image segmentation algorithms.

## Test Images

The `test-images` directory contains three sample images that showcase the results of the implemented algorithms (+ 2 additional images in `/results`):

- `original.jpg`: The original input image.
- `threshold-result.jpg`: Result of the Threshold algorithm.
- `variable-threshold-result.jpg`: Result of the Variable Threshold algorithm.

![apple image segmenetation](https://github.com/nsimona/cuda-threshold/blob/main/test-images/results/apple.gif)

![horse image segmenetation](https://github.com/nsimona/cuda-threshold/blob/main/test-images/results/horse.gif)

![brain image segmenetation](https://github.com/nsimona/cuda-threshold/blob/main/test-images/results/brain.gif)
