#ifndef JPEG_HPP
#define JPEG_HPP

#include "UMI/Image//Graphic.hpp"
#include "UMI/Image/OCV/OCV.hpp"

class ImageJPEG : public OCV, public Graphic {

    public:
        ImageJPEG();
        ImageJPEG(const char* fn);
        virtual ~ImageJPEG();

        void save(const char* path);
    
    
    private:
        // Read the image using OpenCV
        cv::Mat readImage;
        
        void readPixels() override;
        void setWidthHeightChannels();
};

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

#endif