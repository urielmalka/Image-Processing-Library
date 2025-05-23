#pragma once

#ifndef PPM_HPP
#define PPM_HPP

#include <iostream>
#include <memory>

#include "UMI/Image/Graphic.hpp"

enum PPMFormat {
    P1,P2,P3,P4,P5,P6,P7,UNKNOWN
};

PPMFormat getPPMFormat(const char* filename);
unique_ptr<ImagePPM> getPPMTypeClass(const char* filename);


class ImagePPM : public Graphic {

    public:
        ImagePPM(PPMFormat p);
        ImagePPM(const char* fn, PPMFormat p);
        virtual ~ImagePPM();

        PPMFormat ppmFormat;

        /*Filters*/
        void toGray() override;

        virtual void save(const char* path);
    
    protected:
        virtual void readPixels() override;

    private:
        
};


unique_ptr<ImagePPM> getPPMTypeClass(const char* filename);


#endif




