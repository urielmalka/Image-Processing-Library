#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <vector>
#include "../Graphic.hpp"

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


#endif