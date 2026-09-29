cmake_minimum_required(VERSION 3.10)
project(sfm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# compile_commands.json for clangd, symlinked to the source root where editors look for it.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SFM_BUILD_TESTS "Build unit tests" ON)
option(SFM_USE_OPENMP "Parallelize feature extraction and matching" ON)

###########
## Deps  ##
###########

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d flann calib3d)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)

# SIFT lives in core OpenCV from 4.4 on. Older builds need contrib (xfeatures2d),
# otherwise we fall back to the bundled implementation in src/features/sift_impl.cpp.
set(SFM_SIFT_BACKEND "bundled")
if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.4.0")
  set(SFM_SIFT_BACKEND "opencv")
else()
  find_package(OpenCV QUIET COMPONENTS xfeatures2d)
  if(OpenCV_xfeatures2d_FOUND)
    set(SFM_SIFT_BACKEND "xfeatures2d")
  endif()
endif()
message(STATUS "sfm: OpenCV ${OpenCV_VERSION}, Ceres ${Ceres_VERSION}, SIFT backend: ${SFM_SIFT_BACKEND}")

if(SFM_USE_OPENMP)
  find_package(OpenMP)
endif()

###########
## Build ##
###########

file(GLOB_RECURSE SFM_SRCS src/*.cpp)

add_library(sfm ${SFM_SRCS})
target_include_directories(sfm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sfm PUBLIC ${OpenCV_LIBS} Eigen3::Eigen ceres yaml-cpp)
target_compile_options(sfm PRIVATE -Wall -Wextra)

if(SFM_SIFT_BACKEND STREQUAL "opencv")
  target_compile_definitions(sfm PRIVATE SFM_SIFT_OPENCV)
elseif(SFM_SIFT_BACKEND STREQUAL "xfeatures2d")
  target_compile_definitions(sfm PRIVATE SFM_SIFT_XFEATURES2D)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(sfm PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(sfm_main apps/sfm_main.cpp)
target_link_libraries(sfm_main PRIVATE sfm)

###########
## Tests ##
###########

if(SFM_BUILD_TESTS)
  find_package(GTest)
  if(GTest_FOUND OR GTEST_FOUND)
    enable_testing()
    file(GLOB SFM_TEST_SRCS tests/*.cpp)
    add_executable(sfm_tests ${SFM_TEST_SRCS})
    target_link_libraries(sfm_tests PRIVATE sfm GTest::GTest GTest::Main pthread)
    add_test(NAME sfm_tests COMMAND sfm_tests)
  else()
    message(STATUS "sfm: GTest not found, skipping tests")
  endif()
endif()
