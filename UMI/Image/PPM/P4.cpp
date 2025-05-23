#include "P4.hpp"

PPM_P4::PPM_P4(const char* fn): ImagePPM(fn, P4)
{
    channels = P4_CHANNEL_NUM;
    throw std::runtime_error("This library does not support PPM P4 format.\n");
};

PPM_P4::~PPM_P4() {};


void PPM_P4::readPixels(){};

