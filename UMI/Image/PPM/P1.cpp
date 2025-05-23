#include "P1.hpp"

PPM_P1::~PPM_P1() {};


void PPM_P1::readPixels()
{

    data.resize(height, vector<Pixels>(width, BinPixel{0}));

    bool bin;
    
    #pragma omp parallel for
    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            image >> bin;
            data[h][w] = BinPixel{bin};
            cout << bin << endl;
        }
    }
    
    image.close();
};
