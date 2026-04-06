cmake_minimum_required(VERSION 3.18)
project(UMIMAGE VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

# Detect CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND EXISTS "${CMAKE_CUDA_COMPILER}")
    enable_language(CUDA)
    set(HAS_CUDA TRUE)
    message(STATUS "CUDA found: building with GPU support")
else()
    set(HAS_CUDA FALSE)
    message(STATUS "CUDA not found: building without GPU support")
endif()

# Set CUDA architectures explicitly to avoid warning CMP0104
if(HAS_CUDA)
    # Replace with your actual GPU architecture, e.g., 86 for RTX 30XX
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Include directories
include_directories(
    ${PNG_INCLUDE_DIRS}
    ${JPEG_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
)

# Source files
file(GLOB_RECURSE CPP_SOURCES
    "UMI/Image/*.cpp"
    "UMI/Image/Filter/*.cpp"
    "UMI/Image/BMP/*.cpp"
    "UMI/Image/JPEG/*.cpp"
    "UMI/Image/OCV/*.cpp"
    "UMI/Image/PNG/*.cpp"
    "UMI/Image/PPM/*.cpp"
)

if(HAS_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES
        "UMI/Image/Filter/*.cu"
        "UMI/ImageCuda/*.cu"
    )
    list(APPEND CPP_SOURCES ${CUDA_SOURCES})
endif()

# Build target
add_executable(main main.cpp ${CPP_SOURCES})

if(HAS_CUDA)
    target_compile_definitions(main PRIVATE HAS_CUDA)
endif()


set_target_properties(main PROPERTIES LINK_SEARCH_END_STATIC OFF)

target_link_options(main PRIVATE "-Wl,--copy-dt-needed-entries")


# Linker flags
target_link_libraries(main
    OpenMP::OpenMP_CXX
    PNG::PNG
    JPEG::JPEG
    $<$<BOOL:${HAS_CUDA}>:cudart>
)

# CUDA-specific properties
if(HAS_CUDA)
    set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
    )
endif()
