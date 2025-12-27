cmake_minimum_required(VERSION 3.25)

project(pinn VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 3.11 REQUIRED)
find_package(HDF5 2.0 REQUIRED COMPONENTS CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(AddUnitTestExecutable)

add_subdirectory(src)
add_subdirectory(tests)