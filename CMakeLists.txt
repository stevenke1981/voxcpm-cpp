cmake_minimum_required(VERSION 3.16)
project(voxcpm_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---- Options ----
option(VCPM_BUILD_TESTS "Build tests" ON)
option(VCPM_GGML_FETCH "Fetch ggml via FetchContent if not found" ON)

# ---- ggml dependency ----
if(VCPM_GGML_FETCH)
    include(FetchContent)
    FetchContent_Declare(ggml
        GIT_REPOSITORY  https://github.com/ggml-org/ggml.git
        GIT_TAG         v0.15.2
        GIT_SHALLOW     TRUE
    )
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    set(GGML_STATIC ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    # Avoid ggml's own BUILD_TESTING enabling
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ggml)
endif()

# ---- Library ----
add_library(voxcpm STATIC
    src/voxcpm.c
    src/model_loader.c
    src/tokenizer.c
    src/sequence.c
    src/wav.c
    src/ggml_backend.c
    src/minicpm4.c
    src/ralm.c
    src/fsq.c
    src/locenc.c
    src/locdit.c
    src/cfm_solver.c
    src/audio_vae.c
    src/generate.c
)

target_include_directories(voxcpm
    PUBLIC  include
    PRIVATE ${ggml_SOURCE_DIR}/include
)

target_link_libraries(voxcpm
    PUBLIC  ggml
)

# ---- CLI ----
add_executable(voxcpm-c src/main.c)
target_link_libraries(voxcpm-c PRIVATE voxcpm)

# ---- Tests ----
if(VCPM_BUILD_TESTS)
    enable_testing()

    # Helper: private includes for tests
    set(VCPM_TEST_PRIVATE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src)

    add_executable(test_smoke tests/test_smoke.c)
    target_include_directories(test_smoke PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_smoke PRIVATE voxcpm)
    add_test(NAME smoke COMMAND test_smoke)

    add_executable(test_wav tests/test_wav.c)
    target_include_directories(test_wav PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_wav PRIVATE voxcpm)
    add_test(NAME wav COMMAND test_wav)

    add_executable(test_sequence tests/test_sequence.c)
    target_include_directories(test_sequence PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_sequence PRIVATE voxcpm)
    add_test(NAME sequence COMMAND test_sequence)

    add_executable(test_minicpm4 tests/test_minicpm4.c)
    target_include_directories(test_minicpm4 PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_minicpm4 PRIVATE voxcpm)
    add_test(NAME minicpm4 COMMAND test_minicpm4)

    add_executable(test_phase5 tests/test_phase5.c)
    target_include_directories(test_phase5 PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_phase5 PRIVATE voxcpm)
    add_test(NAME phase5 COMMAND test_phase5)

    add_executable(test_model_loader_tensors tests/test_model_loader_tensors.c)
    target_include_directories(test_model_loader_tensors PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_model_loader_tensors PRIVATE voxcpm)
    add_test(NAME model_loader_tensors COMMAND test_model_loader_tensors)
endif()
