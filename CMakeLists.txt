cmake_minimum_required(VERSION 2.8.9)
project(TD)
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "\\.#.*")

add_executable(td ${SOURCES})

target_link_libraries (
  td
  Eigen3::Eigen
  ${CMAKE_THREAD_LIBS_INIT}
  )
