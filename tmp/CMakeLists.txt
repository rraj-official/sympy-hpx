cmake_minimum_required(VERSION 3.18)
project(sympy_hpx_kernel LANGUAGES CXX)

find_package(HPX REQUIRED)          # no COMPONENTS

add_library(${func_name} SHARED kernel.cpp)
target_link_libraries(${func_name} PRIVATE HPX::hpx)   # drop HPX::hpx_init
