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

#endif