cmake_minimum_required(VERSION 3.18)
project(hpx_smoke_test LANGUAGES CXX)

find_package(HPX REQUIRED)

add_executable(hpx_smoke hpx_smoke.cpp)
target_link_libraries(hpx_smoke HPX::hpx)
