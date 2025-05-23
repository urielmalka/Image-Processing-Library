#ifndef P2_HPP
#define P2_HPP

#include "PPM.hpp"

/*
    PPM TYPE P2

*/

#define P2_CHANNEL_NUM 1

class PPM_P2 : public ImagePPM {

    public:
        PPM_P2();
        PPM_P2(const char* fn);
        ~PPM_P2();

    private:
        void readPixels() override;

};

#endif
