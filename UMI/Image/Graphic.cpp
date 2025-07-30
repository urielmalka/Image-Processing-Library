#include "Graphic.hpp"
#include <cassert>


Graphic::Graphic(ImageFormat f) 
{
    format = f;
};

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

void Graphic::toGray()
{
    if (grayscalImage) return;
    grayscalImage = true;
    channels = 1;

    #pragma omp parallel for
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            Pixel &px = data[h][w];

            switch (px.type) {
                case PixelType::RGB: {
                    unsigned char luminance = luminanceFormula(&px.rgb);
                    px.type = PixelType::Grayscale;
                    px.gray = Grayscale{luminance};
                    break;
                }
                case PixelType::BGR: {
                    unsigned char luminance = luminanceFormula(&px.bgr);
                    px.type = PixelType::Grayscale;
                    px.gray = Grayscale{luminance};
                    break;
                }
                default:
                    // Do nothing for Grayscale or Bin
                    break;
            }
        }
    }
}


#ifdef HAS_CUDA

    #include <cuda_runtime.h>
    #include "../ImageCuda/cuGraphic.cuh"

    void Graphic::CudaToGray ()
    {
        std::cout << "Converting image to grayscale using CUDA..." << std::endl;

        if(grayscalImage) return;
        grayscalImage = true;
        channels = 1;

        flat();

        size_t bytes = flatdata.size() * sizeof(Pixel);

        Pixel* h_grays = (Pixel*)malloc(bytes);
        

        assert(flatdata.size() == width * height);

        cuGray(flatdata.data(), h_grays, width, height, bytes);

        #pragma omp parallel for
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = h * width + w;
                data[h][w] = h_grays[idx];
            }
        }

        free(h_grays);
    };

#else

    void Graphic::CudaToGray ()
    {
        std::cerr << "CUDA is not available in this build." << std::endl;
    };

#endif // HAS_CUDA

unsigned char Graphic::luminanceFormula(RGB *rgb)
{   
    return (0.299 * rgb->R) + (0.587 * rgb->G) + (0.114 * rgb->B);
};

unsigned char Graphic::luminanceFormula(BGR *bgr)
{   
    return (0.299 * bgr->r) + (0.587 * bgr->g) + (0.114 * bgr->b);
};

void Graphic::rotate(int degrees){};


void Graphic::crop(int xPos,int yPos, int w_size, int h_size)
{
    if (xPos < 0 || yPos < 0 || xPos + w_size > width || yPos + h_size > height)
        throw std::out_of_range("Crop dimensions out of bounds");


    vector<vector<Pixel>>  new_data;
    new_data.reserve(h_size);

    for(int h=yPos ; h < yPos + h_size ; h++)
    {
            new_data.emplace_back(data[h].begin() + xPos, data[h].begin() + xPos + w_size);
    }
    

    data = move(new_data);
    width = w_size;
    height = h_size;

};

void Graphic::padding(int padding_w, int padding_h)
{
    int pW = padding_w / 2 + (padding_w % 2);
    int pH = padding_h / 2 + (padding_h % 2);
    
    height += pW;
    width += pH;

    vector<vector<Pixel>>  tempData;
    
    tempData.resize(height);

    for (int h = 0; h < height; ++h)
    {
        tempData[h].resize(width);
    }
    
    for (int h = pH; h < height - pH; ++h) {
        for (int w = pW; w < width - pW; ++w) {
            tempData[h][w] = data[h - pH][w - pW];
        }
    }

    data = move(tempData);

}


void Graphic::setObject(Graphic* oldImage)
{
    data = move(oldImage->data);

    type = oldImage->type;

    height = oldImage->height;
    width = oldImage->width;
    max_value = oldImage->max_value;
    channels = oldImage->channels;

    loadFileSuccess = oldImage->loadFileSuccess;
    grayscalImage = oldImage->grayscalImage;
}


void Graphic::setData(vector<vector<Pixel>>  &newData)
{
    height = newData.size();
    width = newData[0].size();
    data = move(newData);
}
