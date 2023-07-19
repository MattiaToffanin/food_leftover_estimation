cmake_minimum_required(VERSION 3.25)
project(food_leftover_estimation)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(food_leftover_estimation cpp_code/main.cpp)
target_link_libraries(food_leftover_estimation ${OpenCV_LIBS})
target_link_libraries(food_leftover_estimation ${PYTHON_LIBRARIES})
