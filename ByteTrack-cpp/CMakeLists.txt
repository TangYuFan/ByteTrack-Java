# ByteTrack CMake 配置（Clang + x86/x64 + JNI DLL）
cmake_minimum_required(VERSION 3.14)
project(bytetrack LANGUAGES CXX)

# -----------------------------
# C++17 标准
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# 构建类型默认 Release
# -----------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# -----------------------------
# 编译器优化与警告
# -----------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -MMD -Wall -Wextra -Winit-self")
endif()

# -----------------------------
# 依赖：Eigen3（header-only）
# -----------------------------
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen-3.3.9")
if(EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen")
  message(STATUS "Found local Eigen3 in ${EIGEN3_INCLUDE_DIR}")
  include_directories(${EIGEN3_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Eigen3 not found in ${EIGEN3_INCLUDE_DIR}")
endif()


# -----------------------------
# 构建 JNI 动态库（Windows64）
# -----------------------------
option(BUILD_JNI_4_WIN "Build JNI library for Windows" OFF)
if(BUILD_JNI_4_WIN)
  # jni 头文件
  include_directories(C:/work/jdk8/include)
  include_directories(C:/work/jdk8/include/win32)
  # bytetrack
  add_library(bytetrack_jni SHARED
          ./bytetrack_jni.cpp
          src/BYTETracker.cpp
          src/KalmanFilter.cpp
          src/lapjv.cpp
          src/Object.cpp
          src/Rect.cpp
          src/STrack.cpp
          )
  target_include_directories(bytetrack_jni PUBLIC
          ${CMAKE_CURRENT_SOURCE_DIR}/include
          )
endif()

# -----------------------------
# 构建 JNI 动态库（android）
# -----------------------------
option(BUILD_JNI_4_ANDROID "Build JNI library for Windows" OFF)
if(BUILD_JNI_4_ANDROID)
endif ()