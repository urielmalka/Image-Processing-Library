#ifndef P4_HPP
#define P4_HPP

#include "../PPM.hpp"

#define P4_CHANNEL_NUM 1

class PPM_P4 : public ImagePPM {

    public:
        PPM_P4(const char* fn);
        ~PPM_P4();

        void readPixels() override;

    private:

};



#endif