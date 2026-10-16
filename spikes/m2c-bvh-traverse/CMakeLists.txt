cmake_minimum_required(VERSION 3.20)
project(m2c_bvh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_executable(m2c_bvh main.cpp)
# The shared headers live in spikes/common (one folder up).
target_include_directories(m2c_bvh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(m2c_bvh PRIVATE Vulkan::Vulkan)
