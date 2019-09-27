cmake_minimum_required(VERSION 3.5)
project(mindreader)

# find_package(imgui CONFIG)

find_package(PkgConfig REQUIRED)

find_package(OpenGL REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
pkg_search_module(GLEW REQUIRED glew)

set(CMAKE_CXX_STANDARD 17)

# add_subdirectory(lib/abseil-cpp)
add_subdirectory(lib/fmt)
add_subdirectory(lib/imgui)


add_executable(mindreader main.cpp util.cpp)

add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLEW)

target_include_directories(mindreader PUBLIC ${GLFW_INCLUDE_DIRS})
target_include_directories(mindreader PUBLIC "lib/imgui")
#target_include_directories(mindreader PUBLIC "imgui")

target_compile_definitions(mindreader PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

target_link_libraries(mindreader
#    absl::strings
    fmt::fmt
    ${GLFW_LIBRARIES}
    ${GLEW_LIBRARIES}
    imgui)
