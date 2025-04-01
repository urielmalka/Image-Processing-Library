#ifndef JPEG_HPP
#define JPEG_HPP

#include "../Graphic.hpp"
#include "../OCV/OCV.hpp"

class ImageJPEG : public OCV, public Graphic {

    public:
        ImageJPEG(const char* fn);
        virtual ~ImageJPEG();

        void save(const char* path);
    
    
    private:
        // Read the image using OpenCV
        cv::Mat readImage;
        
        void readPixels() override;
        void setWidthHeightChannels();
};

#endif