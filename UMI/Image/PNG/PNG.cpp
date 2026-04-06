#include "PNG.hpp"


ImagePNG::ImagePNG () : Graphic(PNG) {};

ImagePNG::ImagePNG (const char* fn) : Graphic(fn,PNG) {

    if(!ocvOpenImage(fn)) return;

    type = "PNG";
 
    setWidthHeightChannels();

    readPixels();

};

ImagePNG::~ImagePNG() {};


void ImagePNG::setWidthHeightChannels() {

    ocvSetWidthHeightChannels(&height, &width, &channels);
}


void ImagePNG::readPixels()
{
    ocvReadPixels(&data, height, width, channels);
}


void ImagePNG::save(const char* path)
{
    ocvSave(path,data, height, width, grayscalImage);
}