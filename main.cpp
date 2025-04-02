#include <UMI/UMI.hpp>

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std::chrono;

int main(int argc, char *argv[])
{   
    auto start = high_resolution_clock::now();
    Image* m = new Image("sample_640×426.bmp");

    Dimensions d = m->image->size();

    cout << d.height << "x" << d.width << "x"<< d.channels << endl;

   //m->image->toGray();

    d = m->image->size();

    cout << d.height << "x" << d.width << "x"<< d.channels << endl;
    
    m->image->save("gun.bmp");

    auto stop = high_resolution_clock::now();


    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: "
            << duration.count() << " microseconds" << endl;


    return 0;
}