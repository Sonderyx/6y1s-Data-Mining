cmake_minimum_required(VERSION 3.15)
project(DataMining)

# Установка параметров сборки
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Включаем детализированные логи сборки
set(CMAKE_VERBOSE_MAKEFILE ON)
set(FETCHCONTENT_QUIET OFF)

# Подключаем FetchContent для загрузки внешних библиотек
include(FetchContent)

# Модуль spdlog для логирования
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
)
FetchContent_MakeAvailable(spdlog)

# Уведомление об успешной загрузке библиотек
message(STATUS "All FetchContent dependencies were successfully built.")

# Добавляем исходные файлы проекта
file(GLOB_RECURSE source
    ./src/Top.cpp
    ./src/*.hpp
)

# Отображаем список файлов
message(STATUS "Source files: ${source}")

# Создаем исполняемый файл
add_executable(${PROJECT_NAME} ${source})

# Линкуем библиотеки Matplot++ и spdlog
target_link_libraries(${PROJECT_NAME} PRIVATE
spdlog::spdlog
)

# Подключаем OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")

# Линкуем OpenCV библиотеки
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})