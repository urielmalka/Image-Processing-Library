#include "JPEG.hpp"


ImageJPEG::ImageJPEG (const char* fn) : Graphic(fn,JPEG) {

    // Read the image using OpenCV
    readImage = cv::imread(fn);
    if (readImage.empty()) {
        std::cerr << "Error: Could not open the image." << std::endl;
    }

    type = "JPEG";
 
    setWidthHeightChannels();

    cout << height << " " << width << " " << channels << std::endl;

    readPixels();

};

ImageJPEG::~ImageJPEG() {};


void ImageJPEG::setWidthHeightChannels() {
    height = static_cast<int>(readImage.rows);
    width = static_cast<int>(readImage.cols);
    channels = readImage.channels();
}


void ImageJPEG::readPixels()
{
    if(channels == 1)
        data.resize(height, vector<Pixels>(width, Grayscale{0}));
    else
        data.resize(height, vector<Pixels>(width, RGB{0,0,0}));


    for (int  h= 0; h < height; h++) {
        if (channels == 1) {
            const unsigned char* rowGrayscale = readImage.ptr<unsigned char>(h);
            for (int w = 0; w < width ; w++) {
                data[h][w] = Grayscale{rowGrayscale[w]};
            }
        } else {
            const cv::Vec3b* rowRGB = readImage.ptr<cv::Vec3b>(h);
            for (int w = 0; w < width; w++) {
                data[h][w] = RGB{rowRGB[w][2], rowRGB[w][1], rowRGB[w][0]};
            }
        }
    }


}


void ImageJPEG::save(const char* path)
{
    
    cv::Mat image(height, width, grayscalImage ? CV_8UC1 : CV_8UC3);

    std::cout << data.size() << "x" << data[0].size() << endl;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            if (grayscalImage) {
                auto pixel = std::get<Grayscale>(data[h][w]);
                image.at<uchar>(h, w) = pixel.I;
            } else {
                auto pixel = std::get<RGB>(data[h][w]);
                image.at<cv::Vec3b>(h, w) = cv::Vec3b(pixel.B, pixel.G, pixel.R); // OpenCV uses BGR
            }
        }
    }
    
    // Save as JPEG
    if (cv::imwrite(path, image)) {
        std::cout << "Image saved as " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
    }
}