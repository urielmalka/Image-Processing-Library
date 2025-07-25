#ifndef P5_HPP
#define P5_HPP

#include "PPM.hpp"

/*
    PPM TYPE P5

*/

#define P5_CHANNEL_NUM 1

class PPM_P5 : public ImagePPM {

    public:
        PPM_P5(const char* fn);
        ~PPM_P5();

    private:
        void readPixels() override;

};


#endif
