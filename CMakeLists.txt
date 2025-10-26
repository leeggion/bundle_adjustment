cmake_minimum_required(VERSION 3.10)
project(BundleAdjustment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} include)

add_executable(bundle
    src/main.cpp
)

target_link_libraries(bundle ${OpenCV_LIBS})
