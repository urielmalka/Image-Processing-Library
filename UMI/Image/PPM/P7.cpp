#include "P7.hpp"  

PPM_P7::PPM_P7(const char* fn): ImagePPM(fn, P7)
{
    channels = P7_CHANNEL_NUM;
    throw runtime_error("This library does not support PPM P7 format.\n");
};

PPM_P7::~PPM_P7() {};


void PPM_P7::readPixels(){};