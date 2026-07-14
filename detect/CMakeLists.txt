cmake_minimum_required(VERSION 3.20)

project(rm_stage2_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(rm_detect
    src/main.cpp
    src/input_handler.cpp
    src/detector.cpp
    src/visualizer.cpp
    src/timer.cpp
)

target_include_directories(rm_detect
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rm_detect
    PRIVATE
    ${OpenCV_LIBS}
)