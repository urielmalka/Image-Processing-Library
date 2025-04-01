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

    data.resize(height, vector<Pixels>(width, Grayscale{0}));

    int i;
    
    #pragma omp parallel for
    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            image >> i;
            cout << i << endl;
            data[h][w] = Grayscale{static_cast<unsigned char>(i)};
        }
    }


    image.close(); // Close file Image
};
