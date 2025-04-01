#ifndef GRAPHIC_HPP
#define GRAPHIC_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <variant>
#include <vector>
#include <algorithm> 
#include <memory>
#include <string>
#include <omp.h>

class ImagePPM;
class ImageJPEG;
class ImagePNG;
class ImageBMP;

using namespace std;

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


using Pixels = variant<RGB, BGR, Grayscale, BinPixel>;

class Graphic {

    public:

        Graphic(const char* fn, ImageFormat f);
        virtual ~Graphic(); 

        ImageFormat format;

        string filename;
        fstream image;
        string type;

        int height;
        int width;
        int max_value;
        int channels;

        bool loadFileSuccess = false; 
        bool grayscalImage = false;

        vector<vector<Pixels>>  data;
        vector<Pixels> flatdata;

        void flat();
        virtual void toGray();
        virtual void save(const char* path);

        Dimensions size(){ return Dimensions{height, width, channels}; }
    
        protected:
            bool openImage();
            bool openImageBinary();

            virtual void readPixels();
            unsigned char luminanceFormula(RGB *rgb);

};

#endif