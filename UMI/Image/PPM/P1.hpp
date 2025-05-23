#ifndef P1_HPP
#define P1_HPP

#include "PPM.hpp"

/*
    PPM TYPE P1

*/

#define P1_CHANNEL_NUM 1

class PPM_P1 : public ImagePPM {

    public:
        PPM_P1(const char* fn);
        ~PPM_P1();

    private:
        void readPixels() override;

};


PPM_P1::PPM_P1(const char* fn): ImagePPM(fn, P1)
{
    channels = P1_CHANNEL_NUM;
    max_value = 1;
    readPixels();
    loadFileSuccess = true;
};



#endif