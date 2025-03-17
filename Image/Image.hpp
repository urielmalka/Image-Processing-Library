#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "Graphic.hpp"
#include "PPM/PPM.hpp"
#include "JPEG/JPEG.hpp"

class Image
{
    private:

        ImageFormat getType(const char* filename);
        unique_ptr<Graphic> loadImage(const char* filename);
        

    public:
        Image(const char* filename);
        ~Image();

        unique_ptr<Graphic> image;

        bool cuda_availble;
};


#endif