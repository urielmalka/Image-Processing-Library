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
    UMI/Image/Filter/FilterCuda.cu
)
target_include_directories(main PRIVATE .)
target_link_libraries(main ${OpenCV_LIBS} cudart OpenMP::OpenMP_CXX)