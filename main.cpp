#include <UMI/UMI.hpp>

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std::chrono;

int main(int argc, char *argv[])
{   
    auto start = high_resolution_clock::now();

    UMImage* m = new UMImage("sample.ppm"); // Load image

    std::vector<std::vector<int>> filter_example = {{-2,0,-2},{-1,0,-1},{-1,0,1}}; // Filter 


    m->filter(filter_example); // Applying a filter to the image
    m->convert(PNG); // Convert to new format 
    m->save("edit_sample.png"); // save the image



    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Time taken by function: "<< duration.count() << " milliseconds" << endl;


    return 0;
}