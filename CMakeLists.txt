cmake_minimum_required(VERSION 3.10)
project(UMIMAGE VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
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
    Image/JPEG/JPEG.hpp
    Image/JPEG/JPEG.cpp
)

target_link_libraries(main ${OpenCV_LIBS} cudart)