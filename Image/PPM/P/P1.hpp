#ifndef P1_HPP
#define P1_HPP

#include "../PPM.hpp"

#define P1_CHANNEL_NUM 1

class PPM_P1 : public ImagePPM {

    public:
        PPM_P1(const char* fn);
        ~PPM_P1();

    private:
        void readPixels() override;

};



#endif