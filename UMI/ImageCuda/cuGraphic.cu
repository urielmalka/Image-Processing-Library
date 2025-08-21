#include "cuGraphic.cuh"
#include <iostream>

__device__ Grayscale luminanceFormula(const RGB& color) {
    Grayscale g;
    g.I = static_cast<unsigned char>(
        0.299f * color.R +
        0.587f * color.G +
        0.114f * color.B
    );
    return g;
}

__device__ Grayscale luminanceFormula(const BGR& color) {
    Grayscale g;
    g.I = static_cast<unsigned char>(
        0.299f * color.r +
        0.587f * color.g +
        0.114f * color.b
    );
    return g;
}

__global__ void cuGrayK(Pixel* pixels_color, Pixel* pixels_gray, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    
    if (x >= w || y >= h) return;


    int idx = y * w + x;

    
    Pixel input = pixels_color[idx];
   
    Grayscale result;

    switch (input.type) {
        case PixelType::RGB:
            result = luminanceFormula(input.rgb);
            break;
        case PixelType::BGR:
            result = luminanceFormula(input.bgr);
            break;
        case PixelType::Grayscale:
            result = input.gray;
            break;
        default:
            result = Grayscale{0};
    }

    pixels_gray[idx].type = PixelType::Grayscale;
    pixels_gray[idx].gray = result;

}


void cuGray(Pixel* pixels, Pixel* h_grays, int w, int h, size_t bytes) {
    
    

    Pixel *d_pixels, *d_grays;

    cudaMalloc(&d_grays, bytes);
    cudaMalloc(&d_pixels, bytes);

    cudaMemcpy(d_pixels, pixels, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 numBlocks((w + 15) / 16, (h + 15) / 16);
   
    cuGrayK<<<numBlocks, threadsPerBlock>>>(d_pixels, d_grays, w, h);
    cudaDeviceSynchronize(); 
 
    cudaMemcpy(h_grays, d_grays, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_grays);
    cudaFree(d_pixels);
}

