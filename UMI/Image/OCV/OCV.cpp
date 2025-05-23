#include  "OCV.hpp"

OCV::OCV(){}

OCV::~OCV(){}

void OCV::ocvReadPixels(vector<vector<Pixels>> * data, int height, int width, int channels)
{
    if(channels == 1)
        data->resize(height, vector<Pixels>(width, Grayscale{0}));
    else
        data->resize(height, vector<Pixels>(width, RGB{0,0,0}));


    #pragma omp parallel for
    for (int  h= 0; h < height; h++) {

        if (channels == 1) {
            const unsigned char* rowGrayscale = readImage.ptr<unsigned char>(h);
            for (int w = 0; w < width ; w++) {
                (*data)[h][w] = Grayscale{rowGrayscale[w]};
            }
        } else {
            const cv::Vec3b* rowRGB = readImage.ptr<cv::Vec3b>(h);
            for (int w = 0; w < width; w++) {
                (*data)[h][w] = RGB{rowRGB[w][2], rowRGB[w][1], rowRGB[w][0]};
            }
        }
    }
}

void OCV::ocvSave(const char * path, const vector<vector<Pixels>> & data, int height, int width, bool isGrayscal)
{
    Mat image(height, width, isGrayscal ? CV_8UC1 : CV_8UC3);

    #pragma omp parallel for
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            if (isGrayscal) {
                auto pixel = get<Grayscale>(data[h][w]);
                image.at<uchar>(h, w) = pixel.I;
            } else {
                auto pixel = get<RGB>(data[h][w]);
                image.at<cv::Vec3b>(h, w) = cv::Vec3b(pixel.B, pixel.G, pixel.R); // OpenCV uses BGR
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