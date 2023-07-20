cmake_minimum_required(VERSION 3.25)
project(food_leftover_estimation)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(food_leftover_estimation cpp_code/main.cpp cpp_code/include/find_food.hpp cpp_code/include/segmentation.hpp cpp_code/src/find_food.cpp cpp_code/src/main_helper.cpp cpp_code/src/segmentation.cpp cpp_code/src/multi_classify.cpp cpp_code/include/multi_classify.hpp)
target_link_libraries(food_leftover_estimation ${OpenCV_LIBS})
target_link_libraries(food_leftover_estimation ${PYTHON_LIBRARIES})
