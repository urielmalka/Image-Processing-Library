#ifndef JPEG_HPP
#define JPEG_HPP

#include "../Graphic.hpp"
#include <opencv2/opencv.hpp>

class ImageJPEG : public Graphic {

    public:
        ImageJPEG(const char* fn);
        virtual ~ImageJPEG();

        void readPixels() override;


        void save(const char* path);
    
    
    private:
        // Read the image using OpenCV
        cv::Mat readImage;

        void setWidthHeightChannels();
};

#endif