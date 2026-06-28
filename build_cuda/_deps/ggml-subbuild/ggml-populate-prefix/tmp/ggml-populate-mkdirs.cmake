# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/voxcpm-cpp/build_cuda/_deps/ggml-src")
  file(MAKE_DIRECTORY "D:/voxcpm-cpp/build_cuda/_deps/ggml-src")
endif()
file(MAKE_DIRECTORY
  "D:/voxcpm-cpp/build_cuda/_deps/ggml-build"
  "D:/voxcpm-cpp/build_cuda/_deps/ggml-subbuild/ggml-populate-prefix"
  "D:/voxcpm-cpp/build_cuda/_deps/ggml-subbuild/ggml-populate-prefix/tmp"
  "D:/voxcpm-cpp/build_cuda/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp"
  "D:/voxcpm-cpp/build_cuda/_deps/ggml-subbuild/ggml-populate-prefix/src"
  "D:/voxcpm-cpp/build_cuda/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/voxcpm-cpp/build_cuda/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/voxcpm-cpp/build_cuda/_deps/ggml-subbuild/ggml-populate-prefix/src/ggml-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
