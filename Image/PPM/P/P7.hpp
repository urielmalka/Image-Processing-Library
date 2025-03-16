#ifndef P7_HPP
#define P7_HPP

#include "../PPM.hpp"

#define P7_CHANNEL_NUM 1

class PPM_P7 : public ImagePPM {

    public:
        PPM_P7(const char* fn);
        ~PPM_P7();

        void readPixels() override;

    private:

};



#endif