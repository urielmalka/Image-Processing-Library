#ifndef BMP_HPP
#define BMP_HPP

#include "UMI/Image/Graphic.hpp"

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature = 0x4D42;  // 'BM'
    uint32_t fileSize;
    uint16_t reserved1 = 0;
    uint16_t reserved2 = 0;
    uint32_t dataOffset = 54;    // Header size for 24-bit BMP
};

struct DIBHeader {
    uint32_t headerSize = 40;
    int32_t width;
    int32_t height;
    uint16_t planes = 1;
    uint16_t bitsPerPixel = 24;
    uint32_t compression = 0;
    uint32_t imageSize;
    int32_t xPixelsPerMeter = 2835;
    int32_t yPixelsPerMeter = 2835;
    uint32_t colorsUsed = 0;
    uint32_t importantColors = 0;
};

class ImageBMP : public Graphic
{
    private:
        void readPixels() override;

    public:
        ImageBMP();
        ImageBMP(const char* fn);
        ~ImageBMP();

        void save(const char* path);
};



#endif