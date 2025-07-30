#ifdef __CUDACC__
    #include <cuda_runtime.h>

    // cuGraphic.cuh
    #ifndef CUGRAPHIC_CUH
    #define CUGRAPHIC_CUH

    #include "../Image/GraphicTypes.hpp"

    // Kernel function
    void cuGray(Pixel* pixels, Pixel* h_grays, int w, int h, size_t bytes);

    #endif // CUGRAPHIC_CUH

#else

    inline void cuGray(Pixel* pixels, Pixel* h_grays, int w, int h, size_t bytes);

#endif
