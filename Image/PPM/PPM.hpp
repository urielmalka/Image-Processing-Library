#ifndef PPM_HPP
#define PPM_HPP

#include "../Graphic.hpp"

class PPM_P1;
class PPM_P2;
class PPM_P3;
class PPM_P4;
class PPM_P5;
class PPM_P6;
class PPM_P7;

enum PPMFormat {
    P1,P2,P3,P4,P5,P6,P7,UNKNOWN
};

PPMFormat getPPMFormat(const char* filename);
unique_ptr<ImagePPM> getPPMTypeClass(const char* filename);


class ImagePPM : public Graphic {

    public:
        ImagePPM(const char* fn, PPMFormat p);
        virtual ~ImagePPM();

        PPMFormat ppmFormat;

        virtual void readPixels() override;

        /*Filters*/
        void toGray() override;

        void save(const char* path);
    
    
    private:
        
};


#endif


