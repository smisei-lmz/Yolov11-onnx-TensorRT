cmake_minimum_required(VERSION 3.22)
project(tensorrt_cpp_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TENSORRT_ROOT "/home/smisei/opt/tensorrt/TensorRT-11.3.0.99")
set(CUDAToolkit_ROOT "/usr/local/cuda-13.4")

find_package(CUDAToolkit 13.4 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS "${TENSORRT_ROOT}/include" NO_DEFAULT_PATH)
find_library(TENSORRT_LIBRARY nvinfer PATHS "${TENSORRT_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or libnvinfer.so not found in ${TENSORRT_ROOT}")
endif()

add_executable(trt_example
    main.cpp
    src/BufferManager.cpp
    src/Preprocess.cpp
    src/Postprocess.cpp
    src/TrtEngine.cpp
)
target_include_directories(trt_example PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${TENSORRT_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(trt_example PRIVATE
    "${TENSORRT_LIBRARY}"
    CUDA::cudart
    ${OpenCV_LIBS}
)
set_target_properties(trt_example PROPERTIES BUILD_RPATH "${TENSORRT_ROOT}/lib")
