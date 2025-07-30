
#ifndef GRAPHIC_TYPES_HPP
#define GRAPHIC_TYPES_HPP

#include <cstdint>

enum ImageFormat {
    PPM,
    PNG,
    JPEG,
    BMP,
    GIF,
    TIFF,
    WEBP,
    HEIF,
    AVIF,
    ERROR_FORMAT
};

enum class PixelType {
    RGB,
    BGR,
    Grayscale,
    Bin
};


struct RGB{
    unsigned char R,G,B;
};

struct BGR {
    uint8_t b, g, r; // for .BMP
};

struct Grayscale{
    unsigned char I; // intensity
};

struct BinPixel{
    bool BIN; // Binary value 
};

struct Dimensions{
    int height;
    int width;
    int channels;
};


struct Pixel {
    PixelType type;

    union {
        RGB rgb;
        BGR bgr;
        Grayscale gray;
        BinPixel bin;
    };

    Pixel() = default;

    Pixel(RGB value) : type(PixelType::RGB), rgb(value) {}
    Pixel(BGR value) : type(PixelType::BGR), bgr(value) {}
    Pixel(Grayscale value) : type(PixelType::Grayscale), gray(value) {}
    Pixel(BinPixel value) : type(PixelType::Bin), bin(value) {}
};

#endif
