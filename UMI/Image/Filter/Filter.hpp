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
        Pixels calculation_filter(vector<vector<Pixels>> &image_pixels , const vector<PixelLocation> &work_pixels);
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

        vector<vector<Pixels>> make_filter(vector<vector<Pixels>> &image_pixels);
};



Filter::Filter(vector<vector<int>> f) 
{
    filter = move(f);
    height = filter.size();
    width = filter[0].size();


    _hf = height / 2;
    _wf = width / 2;

    h_isOdd = height % 2;
    w_isOdd = width % 2;
    
    divid_by =0;
    for (const auto& row : f) {
        for (const int& num : row) {
            divid_by += num;
        }
    }
};

Filter::Filter(vector<vector<float>> f) 
{
    //filter = move(f);
    height = filter.size();
    width = filter[0].size();


    _hf = height / 2;
    _wf = width / 2;

    h_isOdd = height % 2;
    w_isOdd = width % 2;
    
    divid_by =0; // Not use for this constractor because the filter is float type
};

Filter::~Filter(){};

vector<vector<Pixels>> Filter::make_filter(vector<vector<Pixels>> &image_pixels)
{
    int image_h = image_pixels.size();
    int image_w = image_pixels[0].size();

    std::cout << _hf << "|" << _wf << endl;

    vector<PixelLocation> locations;
    vector<vector<Pixels>> image(image_h - _hf - h_isOdd, vector<Pixels>(image_w - _wf - w_isOdd));

    std::cout << image.size() << "|" << image[0].size() << endl;

    for(int h = _hf ; h < (image_h - _hf)  ; h++){
        for(int w = _wf ; w < (image_w - _wf)  ; w++){
            locations = getFilterPixelLocations(h,w);
            image[h - _hf][w - _wf] = calculation_filter(image_pixels,locations);
        }
    }

    return image;
}

Pixels Filter::calculation_filter(vector<vector<Pixels>> &image_pixels , const vector<PixelLocation> &work_pixels)
{

    int r =0,g=0,b=0;

    for(const PixelLocation &wp : work_pixels)
    {

        if(auto *rgb = get_if<RGB>(&image_pixels[wp.y][wp.x]))
        {
            r += (rgb->R * wp.f);
            g += (rgb->G * wp.f);
            b += (rgb->B * wp.f);
        }

    }

    if(divid_by >0)
    {
        r = r / divid_by;
        g = g / divid_by;
        b = b / divid_by;
    }

    return RGB{static_cast<unsigned char>(r),static_cast<unsigned char>(g),static_cast<unsigned char>(b)};
}


vector<PixelLocation> Filter::getFilterPixelLocations(int y, int x)
{

    vector<PixelLocation> locations;

    locations.reserve(height * width);

    for(int h= - _hf; h <= _hf ; h++){
        for(int w=-_wf; w <= _wf ; w++){
            locations.push_back(PixelLocation{y+h,x+w, filter[h+_hf][w+_wf]});
        }   
    }
    
    return locations;
}

#endif