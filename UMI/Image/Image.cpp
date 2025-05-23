
#include "Image.hpp"


UMImage::UMImage(const char* filename){
    image = loadImage(filename);
};

UMImage::~UMImage(){};

ImageFormat UMImage::getType(const char* filename)
{
    string fname(filename), typeImage;
    size_t pos = fname.rfind('.');
    
    typeImage = fname.substr(pos + 1);

    transform(typeImage.begin(),typeImage.end(), typeImage.begin(), ::toupper);

    cudaGetDeviceCount(&cuda_available);

    FilterCuda *f = new FilterCuda();

    if(typeImage == "PPM") return PPM;
    else if(typeImage == "PNG") return PNG;
    else if (typeImage == "JPEG" || typeImage == "JPG") return JPEG;
    else if (typeImage == "BMP") return BMP;
    else return ERROR_FORMAT; // Error Type 
}

unique_ptr<Graphic> UMImage::loadImage(const char* filename)
{

    ImageFormat tf = getType(filename);

    switch (tf)
    {
        case PPM:
            return getPPMTypeClass(filename);
        case PNG:
             return make_unique<ImagePNG>(filename);
        case JPEG:
            return make_unique<ImageJPEG>(filename);
        case BMP:
            return make_unique<ImageBMP>(filename);
        default:
            return nullptr; // This is error 
    }


};

void filter(const vector<vector<int>> &filterMatrix ,int strides)
{}


void UMImage::filter(const vector<vector<int>> &filterMatrix)
{
    /*if(cuda_available)
    {

    }else{
        
    }*/

    if(filterMatrix.size() == 0 || filterMatrix[0].size() == 0) return;

    padding(filterMatrix.size(),filterMatrix[0].size());

    Filter *filter = new Filter(filterMatrix);
    vector<vector<Pixels>> newData;
    newData = filter->make_filter(image->data);
    image->setData(newData);

}

void UMImage::convert(ImageFormat newFormat)
{
    std::unique_ptr<Graphic> newImage;

    switch (newFormat)
    {
        case PPM:
            if (image->grayscalImage) {
                newImage = std::make_unique<PPM_P2>();
            } else {
                newImage = std::make_unique<PPM_P3>();
            }
            break;
        case PNG:
            newImage = std::make_unique<ImagePNG>();
            break;
        case JPEG:
            newImage = std::make_unique<ImageJPEG>();
            break;
        case BMP:
            newImage = std::make_unique<ImageBMP>();
            break;
        default:
            return; // ERROR
    }

    newImage->setObject(image.get()); 

    image = std::move(newImage); 
}
