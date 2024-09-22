cmake_minimum_required(VERSION 3.15)
project(1_Clustering)

set(CMAKE_CXX_STANDARD 17)           # Использовать стандарт C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Требовать C++17

# Добавляем файлы
file(GLOB_RECURSE source
./src/Top.cpp
./src/*.hpp
)

# Подключаем модуль FetchContent
include(FetchContent)

# FetchContent для nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2 # Укажи актуальную версию
)

# FetchContent для spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.11.0  # Укажи актуальную версию
)

FetchContent_MakeAvailable(json spdlog)

add_executable(${PROJECT_NAME} ${source})

# OpenCV
# set(OpenCV_DIR C:/work/libraries/opencv/build)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV include directories:" ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
nlohmann_json::nlohmann_json
spdlog::spdlog
)