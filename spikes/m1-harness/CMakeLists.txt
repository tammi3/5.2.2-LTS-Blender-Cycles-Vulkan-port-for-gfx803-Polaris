cmake_minimum_required(VERSION 3.20)
project(m1_harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Finds the Vulkan SDK via the VULKAN_SDK environment variable the installer set.
find_package(Vulkan REQUIRED)

add_executable(m1_harness main.cpp)
target_link_libraries(m1_harness PRIVATE Vulkan::Vulkan)
