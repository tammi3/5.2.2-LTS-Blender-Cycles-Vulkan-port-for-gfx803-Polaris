cmake_minimum_required(VERSION 3.20)
project(bvh_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only helpers (obj_loader.h, bvh.h, cpu_ref.h) live next to main.cpp,
# so there is nothing else to configure. No Vulkan needed yet.
add_executable(bvh_test main.cpp)
