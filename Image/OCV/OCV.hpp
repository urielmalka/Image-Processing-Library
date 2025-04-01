#ifndef OCV_HPP
#define OCV_HPP

#include <opencv2/opencv.hpp>
#include "../Graphic.hpp"

using namespace cv;

class OCV
{
    private:
        /* data */

    protected:
        // Read the image using OpenCV
        Mat readImage;

        void ocvReadPixels(vector<vector<Pixels>> * data, int height, int width, int channels);
        void ocvSave(const char * path, const vector<vector<Pixels>> & data, int height, int width, bool isGrayscal);
        void ocvSetWidthHeightChannels(int *width, int *heiger, int *channels);

        bool ocvOpenImage(const char *fn);
    
    public:
        OCV();
        ~OCV();
    };


#endif
