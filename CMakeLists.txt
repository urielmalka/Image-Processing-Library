cmake_minimum_required(VERSION 3.10)
project(UMIMAGE VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE SOURCE_FILES
    "UMI/Image/*.cpp"
    "UMI/Image/Filter/*.cpp"
    "UMI/Image/Filter/*.cu"
    "UMI/Image/BMP/*.cpp"
    "UMI/Image/JPEG/*.cpp"
    "UMI/Image/OCV/*.cpp"
    "UMI/Image/PNG/*.cpp"
    "UMI/Image/PPM/*.cpp"
)


add_executable(main
    main.cpp
    ${SOURCE_FILES}
)

target_include_directories(main PRIVATE .)
target_link_libraries(main ${OpenCV_LIBS} cudart OpenMP::OpenMP_CXX)