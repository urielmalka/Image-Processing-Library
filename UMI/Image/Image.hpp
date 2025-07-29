#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "UMI/Image/Graphic.hpp"
#include "UMI/Image/PPM/PPM.hpp"
#include "UMI/Image/JPEG/JPEG.hpp"
#include "UMI/Image/PNG/PNG.hpp"
#include "UMI/Image/BMP/BMP.hpp"
#include "Filter/Filter.hpp"
#include "Filter/FilterCuda.cuh"


class UMImage
{
    private:

        ImageFormat getType(const char* filename);
        unique_ptr<Graphic> loadImage(const char* filename);
        unique_ptr<Graphic> image;

    public:
        UMImage(const char* filename);
        ~UMImage();

        

        int cuda_available;

        void convert(ImageFormat newFormat);
        void padding(int w, int h){ image->padding(w, h); };

        void filter(const vector<vector<int>> &filterMatrix ,int strides);
        void filter(const vector<vector<int>> &filterMatrix);

        void rotate(int degrees){ image->rotate(degrees); }
        void toGray();
        void save(const char* path){ image->save(path); }
        void crop(int x,int y, int w, int h) { image->crop(x,y,w,h); };


        Dimensions size(){ return image->size(); }; 
};



#endif