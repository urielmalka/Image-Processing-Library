#pragma once

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

#include "GraphicTypes.hpp"

class ImagePPM;
class ImageJPEG;
class ImagePNG;
class ImageBMP;

using namespace std;



class Graphic {

    public:

        Graphic(ImageFormat f);
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

        vector<vector<Pixel>>  data;
        vector<Pixel> flatdata;

        void padding(int w, int h);
        
        virtual void toGray();
        virtual void CudaToGray();
        virtual void save(const char* path);

        void rotate(int degrees);
        void crop(int x,int y, int w, int h);
        void flat();

        void setObject(Graphic* oldImage);
        void setData(vector<vector<Pixel>>  &newData);

        Dimensions size(){ return Dimensions{height, width, channels}; }
    
        protected:
            bool openImage();
            bool openImageBinary();

            virtual void readPixels();
            unsigned char luminanceFormula(RGB *rgb);
            unsigned char luminanceFormula(BGR *bgr);

};

#endif