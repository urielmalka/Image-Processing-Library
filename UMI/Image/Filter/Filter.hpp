#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <vector>
#include "UMI/Image/Graphic.hpp"

using namespace std;

class Filter
{
    private:
        void calculation_filter(vector<vector<int>> &image_pixels , const vector<vector<int>> &filter , int x, int y);

        int width;
        int height;
        Pixels typePixels;

        vector<vector<Pixels>> image;
        
    public:
        Filter(int w, int h, Pixels tPixels);
        ~Filter();

        void make_filter(vector<vector<int>> &image_pixels , const vector<vector<int>> &filter , int strides);
};



Filter::Filter(int w, int h, Pixels tPixels)
{
    typePixels = tPixels;
    width = w;
    height = h;
    image.resize(width, vector<Pixels>(height,typePixels));
};
Filter::~Filter(){};

void Filter::make_filter(vector<vector<int>> &image_pixels , const vector<vector<int>> &filter , int strides)
{
    
}

void Filter::calculation_filter(vector<vector<int>> &image_pixels , const vector<vector<int>> &filter , int x, int y)
{

    int filter_x,filter_y,max_fx,max_fy, min_fx, min_fy;

    filter_x = 0;
    filter_y = 0;

    max_fx = x + (filter.size() / 2);
    max_fy = y + (filter[x].size() / 2);
    min_fx =  x - (filter.size() / 2);
    min_fy =  y - (filter[x].size() / 2);

    if(min_fx < 0){
        min_fx = 0;
        filter_x ++;
    }
    else if (max_fx > image_pixels.size() - 1) max_fx = image_pixels.size() - 1;
    
    if(min_fy < 0){
        min_fy = 0;
        filter_y ++;
    }
    else if (max_fy > image_pixels[x].size() - 1) max_fy = image_pixels[x].size() - 1;

    for(int fx=x ; fx < x + filter.size() ; fx ++)
    {
        for(int fy=y ; fy < y + filter[x].size() ; fy++)
        {

        }
    }
}

#endif