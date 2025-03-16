#include "P3.hpp"

PPM_P3::PPM_P3(const char* fn): ImagePPM(fn, P3)
{
    channels = P3_CHANNEL_NUM;
    readPixels();
    loadFileSuccess = true;
};

PPM_P3::~PPM_P3() {};

void PPM_P3::readPixels()
{

    data.resize(width, vector<Pixels>(height, RGB{0,0,0}));

    string line;
    istringstream iss(line);
    unsigned char r,g,b;

    for(int w=0; w < width; w++){
        for(int h=0; h < height; h++){

            iss.clear();
            getline(image,line);
            iss.str(line);

            iss >> r >> g >> b;
            data[w][h] = RGB{r,g,b};
        }
    }
    
    image.close();
};