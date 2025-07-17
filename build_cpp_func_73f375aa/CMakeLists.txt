
cmake_minimum_required(VERSION 3.10)
project(sympy_hpx_kernel)
find_package(HPX REQUIRED)
add_library(cpp_func_73f375aa SHARED kernel.cpp)
target_link_libraries(cpp_func_73f375aa PRIVATE HPX::hpx)
