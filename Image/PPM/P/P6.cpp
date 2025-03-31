#include "P6.hpp"

PPM_P6::PPM_P6(const char* fn): ImagePPM(fn, P6)
{
    channels = P6_CHANNEL_NUM;

    readPixels();
    loadFileSuccess = true;
};

PPM_P6::~PPM_P6() {};

void PPM_P6::readPixels()
{

    data.resize(height, vector<Pixels>(width, RGB{0,0,0}));

    string line;
    istringstream iss(line);
    unsigned char r,g,b;
    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            image.read(reinterpret_cast<char *>(&r), 1);
            image.read(reinterpret_cast<char *>(&g), 1);
            image.read(reinterpret_cast<char *>(&b), 1);

            iss >> r >> g >> b;

            data[h][w] = RGB{r,g,b};
        }
    }
    
    image.close();
};