#include "BMP.hpp"


ImageBMP::ImageBMP(const char* fn) : Graphic(fn,BMP)
{

    if(!openImageBinary()) return;

    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    image.read(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));
    image.read(reinterpret_cast<char*>(&dibHeader), sizeof(DIBHeader));

    // Check if it's a BMP file
    if (bmpHeader.signature != 0x4D42) {
        std::cerr << "Not a BMP file!" << std::endl;
    }

    image.seekg(bmpHeader.dataOffset, ios::beg);

    height = dibHeader.height;
    width = dibHeader.width;
    channels = 3;

    readPixels();

    image.close();
}

ImageBMP::~ImageBMP(){}

void ImageBMP::readPixels()
{
    data.resize(height, vector<Pixels>(width, BGR{0,0,0}));

    int padding = (4 - (width * sizeof(BGR)) % 4) %4;

    for(int h = 0; h < height; ++h) {
        int row = height - 1 - h; // read from bottom to top
        image.read(reinterpret_cast<char*>(data[row].data()), width * sizeof(BGR));
        //image.ignore(padding);
    }
};


void ImageBMP::save(const char* path)
{
    const int paddingSize = (4 - (width * 3) % 4) % 4;
    const int rowSize = width * 3 + paddingSize;
    const int pixelDataSize = rowSize * height;
    const int fileSize = 54 + pixelDataSize;

    BMPHeader bmpHeader;
    bmpHeader.fileSize = fileSize;

    DIBHeader dibHeader;
    dibHeader.width = width;
    dibHeader.height = height;
    dibHeader.imageSize = pixelDataSize;

    ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to create output.bmp\n";
        return;
    }

    // Write headers
    out.write(reinterpret_cast<const char*>(&bmpHeader), sizeof(bmpHeader));
    out.write(reinterpret_cast<const char*>(&dibHeader), sizeof(dibHeader));

    for(int h = 0; h < height; ++h) {
        int row = height - 1 - h; // write from bottom to top
        for(int w = 0; w < width; ++w) {
            auto p = data[row][w];
            out.write(reinterpret_cast<const char*>(&p), sizeof(BGR));
        }
        //uint8_t padding[3] = {0, 0, 0};
        //out.write(reinterpret_cast<const char*>(padding), paddingSize);
    }
    

    out.close();
}