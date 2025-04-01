#ifndef P6_HPP
#define P6_HPP

#include "../PPM.hpp"

#define P6_CHANNEL_NUM 3

class PPM_P6 : public ImagePPM {

    public:
        PPM_P6(const char* fn);
        ~PPM_P6();

    private:
        void readPixels() override;

};



#endif