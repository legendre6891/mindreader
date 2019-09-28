find_package(PkgConfig REQUIRED)

find_package(OpenGL REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
pkg_search_module(GLEW REQUIRED glew)





set(IMGUI_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/lib/imgui)

file(GLOB IMGUI_SOURCES ${IMGUI_INCLUDE_DIR}/*.cpp)
file(GLOB IMGUI_HEADERS ${IMGUI_INCLUDE_DIR}/*.h)
    

add_library(imgui STATIC ${IMGUI_HEADERS} ${IMGUI_SOURCES})

target_include_directories(PUBLIC ${GLFW_INCLUDE_DIRS})
