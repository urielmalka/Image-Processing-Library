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

void Graphic::save(const char* path) {};
void Graphic::readPixels() {};


void Graphic::toGray ()
{
    if(grayscalImage) return;
    grayscalImage = true;

    for(int w=0; w < width; w++){
        for(int h=0; h < height; h++){

            if(auto *rgb = get_if<RGB>(&data[w][h])){
                data[w][h] = Grayscale{luminanceFormula(rgb)};
            }
            
        }
    }
};

unsigned char Graphic::luminanceFormula(RGB *rgb)
{   
    return (0.299 * rgb->R) + (0.587 * rgb->G) + (0.114 * rgb->B);
};

