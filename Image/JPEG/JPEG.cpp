#include "JPEG.hpp"


ImageJPEG::ImageJPEG (const char* fn) : Graphic(fn,JPEG) {

    if(!openImage()) {};

    type = "JPEG";
 
    setWidthHeightChannels();

    cout << width << " " << height << " " << channels << std::endl;

    readPixels();

};

ImageJPEG::~ImageJPEG() {};


void ImageJPEG::setWidthHeightChannels() {

    unsigned char buffer[2];
    
    // Read the first two bytes (JPEG header)
    image.read(reinterpret_cast<char *>(buffer), 2);
    if (buffer[0] != 0xFF || buffer[1] != 0xD8) {
        throw runtime_error("Not a valid JPEG file!");
    }

    // Look for the Start of Frame (SOF0 or SOF2) marker
    while (image.read(reinterpret_cast<char *>(buffer), 2)) {
        if (buffer[0] == 0xFF && (buffer[1] >= 0xC0 && buffer[1] <= 0xC3)) { 
            // SOF0 (0xC0) or SOF2 (0xC2) found, which contain image dimensions
            
            unsigned char segmentLength[2];
            image.read(reinterpret_cast<char *>(segmentLength), 2); // Read segment length (not needed here)
            
            unsigned char precision;
            image.read(reinterpret_cast<char *>(&precision), 1); // Read precision (8-bit for standard JPEG)

            unsigned char sizeData[4];
            image.read(reinterpret_cast<char *>(sizeData), 4); // Read height and width

            height = (sizeData[0] << 8) + sizeData[1];
            width = (sizeData[2] << 8) + sizeData[3];

            unsigned char tempChannel;
            image.read(reinterpret_cast<char *>(&tempChannel), 1); // Read number of color channels
            channels = static_cast<int>(tempChannel);

            return;
        } 
        else {  
            // Skip the current segment
            unsigned char segmentLength[2];
            image.read(reinterpret_cast<char *>(segmentLength), 2);
            int length = (segmentLength[0] << 8) + segmentLength[1];
            image.seekg(length - 2, ios::cur); // Move to the next segment
        }
    }

    throw runtime_error("Could not find valid SOF marker in JPEG file!");
}


void ImageJPEG::readPixels()
{
    if(channels == 1)
        data.resize(width, vector<Pixels>(height, Grayscale{0}));
    else
        data.resize(width, vector<Pixels>(height, RGB{0,0,0}));

    unsigned char i,r,g,b;

    for(int w=0; w < width; w++){
        for(int h=0; h < height; h++){

            if(channels == 1){
                image.read(reinterpret_cast<char*>(&i), 1);
                data[w][h] = Grayscale{i};
            }else{
                image.read(reinterpret_cast<char*>(&r), 1);
                image.read(reinterpret_cast<char*>(&g), 1);
                image.read(reinterpret_cast<char*>(&b), 1);
                data[w][h] = RGB{r,g,b};
            }

        }
    }


    image.close(); // Close file Image
}


void ImageJPEG::save(const char* path)
{

}