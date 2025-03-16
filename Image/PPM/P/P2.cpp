#include "P2.hpp"


PPM_P2::PPM_P2(const char* fn): ImagePPM(fn, P2)
{
    channels = P2_CHANNEL_NUM;
    
    readPixels();
    loadFileSuccess = true;
    grayscalImage = true;
};

PPM_P2::~PPM_P2() {};


void PPM_P2::readPixels()
{

    data.resize(width, vector<Pixels>(height, Grayscale{0}));

    int i;

    for(int w=0; w < width; w++){
        for(int h=0; h < height; h++){

            image >> i;
            cout << i << endl;
            data[w][h] = Grayscale{static_cast<unsigned char>(i)};
        }
    }


    image.close(); // Close file Image
};
