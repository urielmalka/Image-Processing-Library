#include  "OCV.hpp"

OCV::OCV(){}

OCV::~OCV(){}

void OCV::ocvReadPixels(vector<vector<Pixel>> * data, int height, int width, int channels)
{
    if(channels == 1)
        data->resize(height, vector<Pixel>(width, Pixel{Grayscale{0}}));
    else
        data->resize(height, vector<Pixel>(width, Pixel{RGB{0,0,0}}));


    #pragma omp parallel for
    for (int  h= 0; h < height; h++) {

        if (channels == 1) {
            const unsigned char* rowGrayscale = readImage.ptr<unsigned char>(h);
            for (int w = 0; w < width ; w++) {
                (*data)[h][w] = Pixel{Grayscale{rowGrayscale[w]}};
            }
        } else {
            const cv::Vec3b* rowRGB = readImage.ptr<cv::Vec3b>(h);
            for (int w = 0; w < width; w++) {
                (*data)[h][w] = Pixel{RGB{rowRGB[w][2], rowRGB[w][1], rowRGB[w][0]}};
            }
        }
    }
}

void OCV::ocvSave(const char * path, const vector<vector<Pixel>> & data, int height, int width, bool isGrayscale)
{
    Mat image(height, width, isGrayscale ? CV_8UC1 : CV_8UC3);

    #pragma omp parallel for
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {

            const Pixel& px = data[h][w];

            if (isGrayscale) {
                if (px.type == PixelType::Grayscale)
                    image.at<uchar>(h, w) = px.gray.I;
                else
                    image.at<uchar>(h, w) = 0; // fallback
            } else {
                if (px.type == PixelType::RGB) {
                    image.at<cv::Vec3b>(h, w) = cv::Vec3b(px.rgb.B, px.rgb.G, px.rgb.R);
                } else if (px.type == PixelType::BGR) {
                    image.at<cv::Vec3b>(h, w) = cv::Vec3b(px.bgr.b, px.bgr.g, px.bgr.r);
                } else {
                    image.at<cv::Vec3b>(h, w) = cv::Vec3b(0, 0, 0); // fallback
                }
            }
        }
    }
    
    // Save as JPEG
    if (cv::imwrite(path, image)) {
        std::cout << "Image saved as " << path << std::endl;
    } else {
        std::cerr << "Failed to save image!" << std::endl;
    }
}

void OCV::ocvSetWidthHeightChannels(int* height, int* width, int* channels)
{
    *height = static_cast<int>(readImage.rows);
    *width = static_cast<int>(readImage.cols);
    *channels = readImage.channels();
}

bool OCV::ocvOpenImage(const char *fn)
{
    // Read the image using OpenCV
    readImage = cv::imread(fn);
    if (readImage.empty()) {
        std::cerr << "Error: Could not open the image." << std::endl;
        return false;
    }

    return true;
}