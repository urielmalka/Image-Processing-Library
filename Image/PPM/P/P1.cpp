#include "P1.hpp"


PPM_P1::PPM_P1(const char* fn): ImagePPM(fn, P1)
{
    channels = P1_CHANNEL_NUM;
    max_value = 1;
    readPixels();
    loadFileSuccess = true;
};

PPM_P1::~PPM_P1() {};


void PPM_P1::readPixels()
{

    data.resize(width, vector<Pixels>(height, BinPixel{0}));

    bool bin;

    for(int w=0; w < width; w++){
        for(int h=0; h < height; h++){

            image >> bin;
            data[w][h] = BinPixel{bin};
            cout << bin << endl;
        }
    }
    
    image.close();
};
