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

class ImagePPM;
class ImageJPEG;

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

struct Grayscale{
    unsigned char I; // intensity
};

struct BinPixel{
    bool BIN; // Binary value 
};


using Pixels = variant<RGB, Grayscale, BinPixel>;

class Graphic {

    public:

        Graphic(const char* fn, ImageFormat f);
        virtual ~Graphic(); 

        ImageFormat format;

        string filename;
        fstream image;
        string type;


        int width;
        int height;
        int max_value;
        int channels;

        bool loadFileSuccess = false; 
        bool grayscalImage = false;

        vector<vector<Pixels>>  data;

        virtual void toGray();
        virtual void save(const char* path);
    
        protected:
            bool openImage();

            virtual void readPixels();
            unsigned char luminanceFormula(RGB *rgb);

};

#endif