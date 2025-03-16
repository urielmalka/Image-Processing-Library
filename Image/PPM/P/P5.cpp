#include "P5.hpp"


PPM_P5::PPM_P5(const char* fn): ImagePPM(fn, P5)
{
    channels = P5_CHANNEL_NUM;
    throw std::runtime_error("This library does not support PPM P5 format.\n");
};

PPM_P5::~PPM_P5() {};


void PPM_P5::readPixels(){};
