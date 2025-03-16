#ifndef JPEG_HPP
#define JPEG_HPP

#include "../Graphic.hpp"

class ImageJPEG : public Graphic {

    public:
        ImageJPEG(const char* fn);
        virtual ~ImageJPEG();

        void readPixels() override;


        void save(const char* path);
    
    
    private:
        void setWidthHeightChannels();
};

#endif