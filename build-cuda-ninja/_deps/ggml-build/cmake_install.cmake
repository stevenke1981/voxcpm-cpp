# Install script for directory: D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/voxcpm_c")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-build/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-build/src/ggml.lib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-cpu.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-alloc.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-backend.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-blas.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-cann.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-cpp.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-cuda.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-opt.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-metal.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-rpc.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-virtgpu.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-sycl.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-vulkan.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-webgpu.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-zendnn.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/ggml-openvino.h"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-src/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-build/src/ggml-base.lib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-build/ggml-config.cmake"
    "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-build/ggml-version.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "D:/voxcpm-cpp/build-cuda-ninja/_deps/ggml-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
