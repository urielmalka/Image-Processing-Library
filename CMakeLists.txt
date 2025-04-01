cmake_minimum_required(VERSION 3.10)
project(UMIMAGE VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(main
    main.cpp
    Image/Graphic.hpp
    Image/Graphic.cpp
    Image/Image.hpp
    Image/Image.cpp
    Image/PPM/PPM.hpp
    Image/PPM/PPM.cpp
    Image/PPM/P/P1.hpp 
    Image/PPM/P/P1.cpp 
    Image/PPM/P/P2.hpp 
    Image/PPM/P/P2.cpp 
    Image/PPM/P/P3.hpp 
    Image/PPM/P/P3.cpp 
    Image/PPM/P/P4.hpp 
    Image/PPM/P/P4.cpp 
    Image/PPM/P/P5.hpp 
    Image/PPM/P/P5.cpp 
    Image/PPM/P/P6.hpp 
    Image/PPM/P/P6.cpp 
    Image/PPM/P/P7.hpp 
    Image/PPM/P/P7.cpp 
    Image/OCV/OCV.hpp
    Image/OCV/OCV.cpp
    Image/PNG/PNG.hpp
    Image/PNG/PNG.cpp
    Image/JPEG/JPEG.hpp
    Image/JPEG/JPEG.cpp
    Image/BMP/BMP.hpp
    Image/BMP/BMP.cpp
)

target_link_libraries(main ${OpenCV_LIBS} cudart OpenMP::OpenMP_CXX)