#include "Graphic.hpp"

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
    flatdata.reserve(height * width);

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

void Graphic::rotate(int degrees){};


void Graphic::crop(int xPos,int yPos, int w_size, int h_size)
{
    if (xPos < 0 || yPos < 0 || xPos + w_size > width || yPos + h_size > height)
        throw std::out_of_range("Crop dimensions out of bounds");


    vector<vector<Pixels>>  new_data;
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

    vector<vector<Pixels>>  tempData;
    
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


void Graphic::setData(vector<vector<Pixels>>  &newData)
{
    height = newData.size();
    width = newData[0].size();
    data = move(newData);
}
