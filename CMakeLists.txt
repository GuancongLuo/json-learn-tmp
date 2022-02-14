cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)
set(DCMAKE_BUILD_TYPE Release)

project(jsonLearn)

include_directories(${PROJECT_SOURCE_DIR}/include)

FILE(GLOB SRC_FILES "src/*.cpp")

add_executable(jsonLearn ${SRC_FILES})

target_link_libraries(jsonLearn)

