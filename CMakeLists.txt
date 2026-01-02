cmake_minimum_required(VERSION 3.25)

project(pinn VERSION 0.1.0 LANGUAGES CXX)

find_package(Catch2 3.11 REQUIRED)
find_package(HDF5 2.0 REQUIRED COMPONENTS CXX)

include(cmake/pinn_build.cmake)
add_subdirectory(src)

include(CTest)
include(Catch)
include(cmake/add_unit_test_executable.cmake)
add_subdirectory(tests)