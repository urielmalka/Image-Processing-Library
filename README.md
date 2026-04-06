# Image-Processing-Library (UMImage)

Small C++17 image processing library + sample app.

## Features

- Load/save images (JPEG/PNG/BMP/PPM)
- Basic operations (e.g. grayscale, crop, rotate, padding)
- Convolution filters (CPU with OpenMP, optional CUDA build)

## Dependencies

- CMake \(\>= 3.18\)
- C++17 compiler
- OpenMP
- libpng
- libjpeg
- Optional: CUDA toolkit (enables `HAS_CUDA`)

Ubuntu/Debian example:

```bash
sudo apt update
sudo apt install -y cmake g++ libpng-dev libjpeg-dev
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

This produces `build/main`.

## Run (sample)

`main.cpp` loads `sample.jpg`, converts it to grayscale, and saves `edit_sample.jpg`.

```bash
./build/main
```

## Minimal usage

```cpp
#include <UMI/UMI.hpp>

int main() {
  UMImage img("sample.jpg");
  img.toGray();
  img.save("edit_sample.jpg");
  return 0;
}
```

## Project layout (high level)

- `UMI/Image/` core image types + format backends
- `UMI/Image/Filter/` filtering implementations (CPU / optional CUDA)
- `main.cpp` example program