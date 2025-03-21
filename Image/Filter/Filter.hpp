#ifndef FILTER_HPP
#define FILTER_HPP

#include <iostream>
#include <vector>

using namespace std;

class Filter
{
    private:
        void calculation_filter(vector<vector<int>> &image_pixels , const vector<vector<int>> &filter , int x, int y);
        
    public:
        Filter();
        ~Filter();

        void make_filter(vector<vector<int>> &image_pixels , const vector<vector<int>> &filter , int strides);
};

Filter::Filter(){}

Filter::~Filter(){}



#endif