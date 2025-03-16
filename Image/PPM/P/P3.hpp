#ifndef P3_HPP
#define P3_HPP

#include "../PPM.hpp"

#define P3_CHANNEL_NUM 3

class PPM_P3 : public ImagePPM {

    public:
        PPM_P3(const char* fn);
        ~PPM_P3();
        
        void readPixels() override;

    private:

};

#endif