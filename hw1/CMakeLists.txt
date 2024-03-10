cmake_minimum_required(VERSION 3.12)
project(hw1)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(${PROJECT_NAME} src/main.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE ${FLAGS})
