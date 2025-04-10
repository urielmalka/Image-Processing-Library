#ifndef PNG_HPP
#define PNG_HPP

#include "UMI/Image/Graphic.hpp"
#include "UMI/Image/OCV/OCV.hpp"

class ImagePNG : public OCV, public Graphic {

    public:
        ImagePNG();
        ImagePNG(const char* fn);
        virtual ~ImagePNG();

        void save(const char* path);
    
    
    private:
        void readPixels() override;
        void setWidthHeightChannels();
};

ImagePNG::ImagePNG () : Graphic(JPEG) {};

ImagePNG::ImagePNG (const char* fn) : Graphic(fn,JPEG) {

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

#endif