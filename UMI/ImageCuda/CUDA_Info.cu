/*

#include <iostream>
#include <cuda_runtime.h>

int main() {
    size_t freeMem, totalMem;

    // Get free and total memory
    cudaMemGetInfo(&freeMem, &totalMem);

    // Convert bytes to megabytes
    std::cout << "GPU Memory Info:\n";
    std::cout << "Free Memory: " << freeMem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Total Memory: " << totalMem / (1024.0 * 1024.0) << " MB\n";

    return 0;
}
*/