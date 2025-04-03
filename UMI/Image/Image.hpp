#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "UMI/Image/Graphic.hpp"
#include "UMI/Image/PPM/PPM.hpp"
#include "UMI/Image/JPEG/JPEG.hpp"
#include "UMI/Image/PNG/PNG.hpp"
#include "UMI/Image/BMP/BMP.hpp"
#include <cuda_runtime.h>

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

        void filter(const vector<vector<int>> &filterMatrix ,int strides);

        void rotate(int degrees){ image->rotate(degrees); }
        void toGray(){ image->toGray(); }
        void save(const char* path){ image->save(path); }
        void crop(int x,int y, int w, int h) { image->crop(x,y,w,h); };
        Dimensions size(){ return image->size(); }; 
};


UMImage::UMImage(const char* filename){
    image = loadImage(filename);
};

UMImage::~UMImage(){};

ImageFormat UMImage::getType(const char* filename)
{
    string fname(filename), typeImage;
    size_t pos = fname.rfind('.');
    
    typeImage = fname.substr(pos + 1);

    transform(typeImage.begin(),typeImage.end(), typeImage.begin(), ::toupper);

    cudaGetDeviceCount(&cuda_available);

    if(typeImage == "PPM") return PPM;
    else if(typeImage == "PNG") return PNG;
    else if (typeImage == "JPEG" || typeImage == "JPG") return JPEG;
    else if (typeImage == "BMP") return BMP;
    else return ERROR_FORMAT; // Error Type 
}

unique_ptr<Graphic> UMImage::loadImage(const char* filename)
{

    ImageFormat tf = getType(filename);

    switch (tf)
    {
        case PPM:
            return getPPMTypeClass(filename);
        case PNG:
             return make_unique<ImagePNG>(filename);
        case JPEG:
            return make_unique<ImageJPEG>(filename);
        case BMP:
            return make_unique<ImageBMP>(filename);
        default:
            return nullptr; // This is error 
    }


};


void UMImage::filter(const vector<vector<int>> &filterMatrix ,int strides)
{
    if(cuda_available)
    {

    }else{
        
    }
}


#endif