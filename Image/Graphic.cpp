#include "Graphic.hpp"
#include "PPM/PPM.hpp"

Graphic::Graphic(const char* fn, ImageFormat f) 
{
    filename = fn;
    format = f;
};

Graphic::~Graphic() {};

bool Graphic::openImage()
{
    image.open(filename);

    if(!image.is_open() ) { std::cerr << "Error: Unable to open file \"" << filename << "\"." << std::endl; return false; };

    return true;
};

bool Graphic::openImageBinary()
{
    image.open(filename, ios::in | ios::binary);

    if(!image.is_open() ) { std::cerr << "Error: Unable to open file \"" << filename << "\"." << std::endl; return false; };

    return true;
};

void Graphic::flat()
{
    flatdata.resize(height * width);

    for (int h=0 ; h < height ; h++)
    {
        for(int w=0; w < width; w++)
        {
            flatdata[w + (h * width)] = data[h][w];
        }
    }
}

void Graphic::save(const char* path) {};
void Graphic::readPixels() {};


void Graphic::toGray ()
{
    if(grayscalImage) return;
    grayscalImage = true;
    channels = 1;
    
    #pragma omp parallel for
    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            if(auto *rgb = get_if<RGB>(&data[h][w])){
                data[h][w] = Grayscale{luminanceFormula(rgb)};
            }else if(auto *bgr = get_if<BGR>(&data[h][w]))
            {
                RGB rgb = RGB{bgr->r, bgr->g, bgr->b};
                data[h][w] = Grayscale{luminanceFormula(&rgb)};
            }
            
        }
    }
};

unsigned char Graphic::luminanceFormula(RGB *rgb)
{   
    return (0.299 * rgb->R) + (0.587 * rgb->G) + (0.114 * rgb->B);
};

