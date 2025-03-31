#include "PPM.hpp"
#include "P/P1.hpp"
#include "P/P2.hpp"
#include "P/P3.hpp"
#include "P/P4.hpp"
#include "P/P5.hpp"
#include "P/P6.hpp"
#include "P/P7.hpp"

ImagePPM::ImagePPM (const char* fn, PPMFormat p) : Graphic(fn,PPM) {

    if(!openImage()) {};
    
    string line;

    ppmFormat = p;

    getline(image,type);

    //if(type != "P3" && type != "P2") { std::cout << "Incorrect format. The format must be P3 but received " << type << "." << std::endl; return; } 
    
    getline(image,line);

    istringstream iss(line);
    iss >> height >> width;
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