#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <vector>
#include "UMI/Image/Graphic.hpp"

using namespace std;

struct PixelLocation
{
    int y;
    int x;
    int f;
};


class Filter
{
    private:
        Pixel calculation_filter(vector<vector<Pixel>> &image_pixels , const vector<PixelLocation> &work_pixels);
        vector<PixelLocation> getFilterPixelLocations(int y, int x);

        int width;
        int height;
        int _hf;
        int _wf;
        int h_isOdd;
        int w_isOdd;
        int divid_by;

        vector<vector<int>> filter;
        
    public:
        Filter(vector<vector<int>> f);
        Filter(vector<vector<float>> f);
        ~Filter();

        vector<vector<Pixel>> make_filter(vector<vector<Pixel>> &image_pixels);
};


#endif