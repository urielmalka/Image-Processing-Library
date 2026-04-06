#ifndef OCV_HPP
#define OCV_HPP

#include "UMI/Image/Graphic.hpp"

class OCV
{
    private:
        /* data */

    protected:
        std::vector<uint8_t> decoded; // interleaved (gray or RGB)
        int decodedWidth = 0;
        int decodedHeight = 0;
        int decodedChannels = 0; // 1 or 3

        void ocvReadPixels(vector<vector<Pixel>> * data, int height, int width, int channels);
        void ocvSave(const char * path, const vector<vector<Pixel>> & data, int height, int width, bool isGrayscale);
        void ocvSetWidthHeightChannels(int *width, int *height, int *channels);

        bool ocvOpenImage(const char *fn);
    
    public:
        OCV();
        ~OCV();
    };





    

#endif
