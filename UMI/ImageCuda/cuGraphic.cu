#include <cuda_runtime.h>
#include "../Image/Graphic.hpp"


__device__ Grayscale luminanceFormula(const RGB& color) {
    Grayscale g;
    g.I = static_cast<unsigned char>(
        0.299f * color.R +
        0.587f * color.G +
        0.114f * color.B
    );
    return g;
}

__global__ void cuGray(RGB* rgb, Grayscale* gray)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;


    printf("[%d, %d, %d]\n",i,j, k);

    gray[i] = luminanceFormula(rgb[j]);

}
