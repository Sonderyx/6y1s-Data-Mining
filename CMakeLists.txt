cmake_minimum_required(VERSION 3.15)
project(1kmeans)

# Добавляем файлы
file(GLOB_RECURSE source
./src/Top.cpp
./src/*.hpp
)

add_executable(${PROJECT_NAME} ${source})

# OpenCV
# set(OpenCV_DIR C:/work/libraries/opencv/build)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV include directories:" ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})