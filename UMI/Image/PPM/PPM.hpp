#pragma once

#ifndef PPM_HPP
#define PPM_HPP

#include <iostream>
#include <memory>

#include "UMI/Image/Graphic.hpp"

enum PPMFormat {
    P1,P2,P3,P4,P5,P6,P7,UNKNOWN
};

PPMFormat getPPMFormat(const char* filename);
unique_ptr<ImagePPM> getPPMTypeClass(const char* filename);


class ImagePPM : public Graphic {

    public:
        ImagePPM(PPMFormat p);
        ImagePPM(const char* fn, PPMFormat p);
        virtual ~ImagePPM();

        PPMFormat ppmFormat;

        /*Filters*/
        void toGray() override;

        virtual void save(const char* path);
    
    protected:
        virtual void readPixels() override;

    private:
        
};

ImagePPM::ImagePPM (PPMFormat p) : Graphic(PPM) 
{
    ppmFormat = p;
}

ImagePPM::ImagePPM (const char* fn, PPMFormat p) : Graphic(fn,PPM) {

    if(!openImage()) return;
    
    string line;

    ppmFormat = p;

    getline(image,type);

    //if(type != "P3" && type != "P2") { std::cout << "Incorrect format. The format must be P3 but received " << type << "." << std::endl; return; } 
    
    getline(image,line);

    istringstream iss(line);
    iss >> width >> height;
    iss.clear();

    if(type != "P1"){
        getline(image,line);
        iss.str(line);
        iss >> max_value;
    }

};

ImagePPM::~ImagePPM() {};
void ImagePPM::readPixels() {};

void ImagePPM::save (const char* path) {

    ofstream save_image;
    save_image.open(path);

    if(save_image.is_open())
    {
        save_image << type << std::endl;
        cout << type << std::endl;

        save_image << height << " " << width << std::endl;
        cout << height << " " << width << std::endl;
        
        if(type != "P1"){
            save_image << max_value << std::endl;
            cout << max_value << std::endl;
        }

        for(int h=0; h < height; h++) 
        {
            for(int w = 0; w < width; w++)
            {
                if(auto *rgb = get_if<RGB>(&data[h][w]))
                    save_image << rgb->R << " " << rgb->G << " " << rgb->B << std::endl;
                else if (auto *gray = get_if<Grayscale>(&data[h][w]))
                {
                    save_image << static_cast<int>(gray->I) << std::endl;
                }else if (auto *bin = get_if<BinPixel>(&data[h][w]))
                {
                    save_image << static_cast<int>(bin->BIN) << std::endl;
                }
                
            }
        }

        save_image.close();
    }
};


void ImagePPM::toGray ()
{
    if(grayscalImage) return;

    type = "P2";
    Graphic::toGray();
};


PPMFormat getPPMFormat(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return UNKNOWN;
    }

    std::string header;
    file >> header;
    file.close();
    
    switch (header[1]) {
        case '1': return P1;
        case '2': return P2;
        case '3': return P3;
        case '4': return P4;
        case '5': return P5;
        case '6': return P6;
        case '7': return P7;
        default: return UNKNOWN;
    }
}


/*
    PPM TYPE P1

*/

#define P1_CHANNEL_NUM 1

class PPM_P1 : public ImagePPM {

    public:
        PPM_P1(const char* fn);
        ~PPM_P1();

    private:
        void readPixels() override;

};


PPM_P1::PPM_P1(const char* fn): ImagePPM(fn, P1)
{
    channels = P1_CHANNEL_NUM;
    max_value = 1;
    readPixels();
    loadFileSuccess = true;
};

PPM_P1::~PPM_P1() {};


void PPM_P1::readPixels()
{

    data.resize(height, vector<Pixels>(width, BinPixel{0}));

    bool bin;
    
    #pragma omp parallel for
    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            image >> bin;
            data[h][w] = BinPixel{bin};
            cout << bin << endl;
        }
    }
    
    image.close();
};

/*
    PPM TYPE P2

*/

#define P2_CHANNEL_NUM 1

class PPM_P2 : public ImagePPM {

    public:
        PPM_P2();
        PPM_P2(const char* fn);
        ~PPM_P2();

    private:
        void readPixels() override;

};

PPM_P2::PPM_P2(): ImagePPM(P2) {};

PPM_P2::PPM_P2(const char* fn): ImagePPM(fn, P2)
{
    channels = P2_CHANNEL_NUM;
    
    readPixels();
    loadFileSuccess = true;
    grayscalImage = true;
};

PPM_P2::~PPM_P2() {};


void PPM_P2::readPixels()
{

    data.resize(height, vector<Pixels>(width, Grayscale{0}));

    int i;
    
    #pragma omp parallel for
    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            image >> i;
            cout << i << endl;
            data[h][w] = Grayscale{static_cast<unsigned char>(i)};
        }
    }


    image.close(); // Close file Image
};



/*
    PPM TYPE P3

*/

#define P3_CHANNEL_NUM 3

class PPM_P3 : public ImagePPM {

    public:
        PPM_P3();
        PPM_P3(const char* fn);
        ~PPM_P3();

    private:
        void readPixels() override;

};

PPM_P3::PPM_P3(): ImagePPM(P3)
{

};

PPM_P3::PPM_P3(const char* fn): ImagePPM(fn, P3)
{
    channels = P3_CHANNEL_NUM;
    readPixels();
    loadFileSuccess = true;
};

PPM_P3::~PPM_P3() {};

void PPM_P3::readPixels()
{

    data.resize(height, vector<Pixels>(width, RGB{0,0,0}));

    string line;
    istringstream iss(line);
    unsigned char r,g,b;
    
    #pragma omp parallel for
    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            iss.clear();
            getline(image,line);
            iss.str(line);

            iss >> r >> g >> b;
            data[h][w] = RGB{r,g,b};
        }
    }
    
    image.close();
};

/*
    PPM TYPE P4

*/

#define P4_CHANNEL_NUM 1

class PPM_P4 : public ImagePPM {

    public:
        PPM_P4(const char* fn);
        ~PPM_P4();

    private:
        void readPixels() override;

};


PPM_P4::PPM_P4(const char* fn): ImagePPM(fn, P4)
{
    channels = P4_CHANNEL_NUM;
    throw std::runtime_error("This library does not support PPM P4 format.\n");
};

PPM_P4::~PPM_P4() {};


void PPM_P4::readPixels(){};



/*
    PPM TYPE P5

*/

#define P5_CHANNEL_NUM 1

class PPM_P5 : public ImagePPM {

    public:
        PPM_P5(const char* fn);
        ~PPM_P5();

    private:
        void readPixels() override;

};


PPM_P5::PPM_P5(const char* fn): ImagePPM(fn, P5)
{
    channels = P5_CHANNEL_NUM;
    throw std::runtime_error("This library does not support PPM P5 format.\n");
};

PPM_P5::~PPM_P5() {};


void PPM_P5::readPixels(){};

/*
    PPM TYPE P6

*/

#define P6_CHANNEL_NUM 3

class PPM_P6 : public ImagePPM {

    public:
        PPM_P6(const char* fn);
        ~PPM_P6();

    private:
        void readPixels() override;
        void save(const char* path) override;

};


PPM_P6::PPM_P6(const char* fn): ImagePPM(fn, P6)
{
    channels = P6_CHANNEL_NUM;

    readPixels();
    loadFileSuccess = true;
};

PPM_P6::~PPM_P6() {};

void PPM_P6::readPixels()
{

    data.resize(height, vector<Pixels>(width, RGB{0,0,0}));

    string line;
    istringstream iss(line);
    unsigned char r,g,b;

    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            image.read(reinterpret_cast<char *>(&r), 1);
            image.read(reinterpret_cast<char *>(&g), 1);
            image.read(reinterpret_cast<char *>(&b), 1);

            iss >> r >> g >> b;

            data[h][w] = RGB{r,g,b};
        }
    }
    
    image.close();
};

void PPM_P6::save(const char* path)
{

    ofstream out(path, ios::binary);
    if (!out.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }

    // Write P6 header
    out << "P6\n" << width << " " << height << "\n255\n";

    vector<RGB> tempFlatData;
    tempFlatData.reserve(height * width * 3);

    // Write binary pixel data
    for (int h = 0; h < height; ++h) {
        for (auto& pixel : data[h]) {
            if (std::holds_alternative<RGB>(pixel)) {
                tempFlatData.push_back(std::get<RGB>(pixel));
            }
            else {
                cerr << "Non-RGB pixel found in data!" << endl;
                // Optional: convert BGR/Grayscale/BinPixel to RGB here if needed
            }
        }
    }

    out.write(reinterpret_cast<const char*>(tempFlatData.data()), width * height * 3);

    out.close();
}


/*
    PPM TYPE P7

*/

#define P7_CHANNEL_NUM 1

class PPM_P7 : public ImagePPM {

    public:
        PPM_P7(const char* fn);
        ~PPM_P7();

    private:
        void readPixels() override;

};


PPM_P7::PPM_P7(const char* fn): ImagePPM(fn, P7)
{
    channels = P7_CHANNEL_NUM;
    throw runtime_error("This library does not support PPM P7 format.\n");
};

PPM_P7::~PPM_P7() {};


void PPM_P7::readPixels(){};

#endif


/* END TYPES */


unique_ptr<ImagePPM> getPPMTypeClass(const char* filename)
{
    switch (getPPMFormat(filename))
    {
        case P1:
            return make_unique<PPM_P1>(filename);
        case P2:
            return make_unique<PPM_P2>(filename);
        case P3:
            return make_unique<PPM_P3>(filename);
        case P4:
            return make_unique<PPM_P4>(filename);;
        case P5:
            return make_unique<PPM_P5>(filename);
        case P6:
            return make_unique<PPM_P6>(filename);
        case P7:
            return make_unique<PPM_P7>(filename);
        default:
            throw std::runtime_error("This library support PPM P1,P2,P3 and P6 format.\n");
    }
}