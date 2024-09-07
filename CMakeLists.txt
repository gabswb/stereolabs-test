cmake_minimum_required(VERSION 3.18)
project(stereolabs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_BUILD_TYPE Debug)

# Add module in charge of finding TensorRT
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(stereolabs src/main.cpp src/TrtEngine.cc src/Yolov10.cpp)

target_link_libraries(stereolabs PUBLIC ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS})# /home/gabriels/dev/libs/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib/libnvonnxparser.so)
target_include_directories(stereolabs PUBLIC include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${${OpenCV_INCLUDE_DIRS}})