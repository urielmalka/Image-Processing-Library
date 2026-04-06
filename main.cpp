#include <UMI/UMI.hpp>

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std::chrono;

int main(int argc, char *argv[])
{   
    auto start = high_resolution_clock::now();

    UMImage* m = new UMImage("sample.jpg"); // Load image

#ifdef HAS_CUDA
    std::cout << "CUDA support: ON (device_count=" << m->cuda_available << ")\n";
#else
    std::cout << "CUDA support: OFF\n";
#endif

    //std::vector<std::vector<int>> filter_example = {{-2,0,-2},{-1,0,-1},{-1,0,1}}; // Filter 

    m->toGray(); // Convert to grayscale

    //m->filter(filter_example); // Applying a filter to the image
    //m->convert(PNG); // Convert to new format 
    m->save("edit_sample.jpg"); // save the image



    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Time taken by function: "<< duration.count() << " milliseconds" << endl;


    return 0;
}