#include "JPEG.hpp"

ImageJPEG::ImageJPEG () : Graphic(JPEG) {}

ImageJPEG::ImageJPEG (const char* fn) : Graphic(fn,JPEG) {

    
    if(!ocvOpenImage(fn)) return;

    type = "JPEG";
 
    setWidthHeightChannels();

    readPixels();

};

ImageJPEG::~ImageJPEG() {};


void ImageJPEG::setWidthHeightChannels() {

    ocvSetWidthHeightChannels(&height, &width, &channels);
}


void ImageJPEG::readPixels()
{
    ocvReadPixels(&data, height, width, channels);
}


void ImageJPEG::save(const char* path)
{
    ocvSave(path,data, height, width, grayscalImage);
}