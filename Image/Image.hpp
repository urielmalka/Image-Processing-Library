#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "Graphic.hpp"
#include "PPM/PPM.hpp"
#include "JPEG/JPEG.hpp"
#include "PNG/PNG.hpp"
#include "BMP/BMP.hpp"
#include <cuda_runtime.h>

class Image
{
    private:

        ImageFormat getType(const char* filename);
        unique_ptr<Graphic> loadImage(const char* filename);
        

    public:
        Image(const char* filename);
        ~Image();

        unique_ptr<Graphic> image;

        int cuda_available;

        void filter(const vector<vector<int>> &filterMatrix ,int strides);
};


#endif