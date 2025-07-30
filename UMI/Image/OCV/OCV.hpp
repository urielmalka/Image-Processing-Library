#ifndef OCV_HPP
#define OCV_HPP

#include <opencv2/opencv.hpp>
#include "UMI/Image/Graphic.hpp"

using namespace cv;

class OCV
{
    private:
        /* data */

    protected:
        // Read the image using OpenCV
        Mat readImage;

        void ocvReadPixels(vector<vector<Pixel>> * data, int height, int width, int channels);
        void ocvSave(const char * path, const vector<vector<Pixel>> & data, int height, int width, bool isGrayscale);
        void ocvSetWidthHeightChannels(int *width, int *height, int *channels);

        bool ocvOpenImage(const char *fn);
    
    public:
        OCV();
        ~OCV();
    };





    

#endif
