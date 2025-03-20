#include "Image.hpp"



Image::Image(const char* filename){
    image = loadImage(filename);
};

Image::~Image(){};

ImageFormat Image::getType(const char* filename)
{
    string fname(filename), typeImage;
    size_t pos = fname.rfind('.');
    
    typeImage = fname.substr(pos + 1);

    transform(typeImage.begin(),typeImage.end(), typeImage.begin(), ::toupper);

    cudaGetDeviceCount(&cuda_available);

    cout << cuda_available << endl;

    if(typeImage == "PPM") return PPM;
    else if(typeImage == "PNG") return PNG;
    else if (typeImage == "JPEG") return JPEG;
    else return ERROR_FORMAT; // Error Type 
}

unique_ptr<Graphic> Image::loadImage(const char* filename)
{

    ImageFormat tf = getType(filename);

    switch (tf)
    {
        case PPM:
            return getPPMTypeClass(filename);
        // case PNG:
        //     return getPPMTypeClass(filename);
        case JPEG:
            return make_unique<ImageJPEG>(filename);
        default:
            return nullptr; // This is error 
    }


};