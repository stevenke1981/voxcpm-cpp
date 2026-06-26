cmake_minimum_required(VERSION 3.16)
project(voxcpm_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---- Options ----
option(VCPM_BUILD_TESTS "Build tests" ON)
set(VCPM_GGML_DIR "" CACHE PATH "Path to an existing ggml checkout")
option(VCPM_GGML_FETCH "Fetch ggml via FetchContent if not found" ON)
option(VCPM_ENABLE_CUDA "Enable ggml CUDA backend" OFF)
option(VCPM_ENABLE_METAL "Enable ggml Metal backend" OFF)
option(VCPM_ENABLE_VULKAN "Enable ggml Vulkan backend" OFF)

# ---- ggml dependency ----
if(VCPM_GGML_DIR)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA ${VCPM_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(GGML_METAL ${VCPM_ENABLE_METAL} CACHE BOOL "" FORCE)
    set(GGML_VULKAN ${VCPM_ENABLE_VULKAN} CACHE BOOL "" FORCE)
    set(GGML_STATIC ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ggml_SOURCE_DIR "${VCPM_GGML_DIR}")
    add_subdirectory("${VCPM_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml")
elseif(VCPM_GGML_FETCH)
    include(FetchContent)
    FetchContent_Declare(ggml
        GIT_REPOSITORY  https://github.com/ggml-org/ggml.git
        GIT_TAG         v0.15.2
        GIT_SHALLOW     TRUE
    )
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA ${VCPM_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(GGML_METAL ${VCPM_ENABLE_METAL} CACHE BOOL "" FORCE)
    set(GGML_VULKAN ${VCPM_ENABLE_VULKAN} CACHE BOOL "" FORCE)
    set(GGML_STATIC ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    # Avoid ggml's own BUILD_TESTING enabling
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ggml)
else()
    find_package(ggml REQUIRED)
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
    src/audio_vae_v2.c
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

    add_executable(test_wav_writer tests/audio/test_wav_writer.c)
    target_include_directories(test_wav_writer PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_wav_writer PRIVATE voxcpm)
    add_test(NAME wav_writer COMMAND test_wav_writer)

    add_executable(test_sequence tests/test_sequence.c)
    target_include_directories(test_sequence PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_sequence PRIVATE voxcpm)
    add_test(NAME sequence COMMAND test_sequence)

    add_executable(test_tokenizer_parity tests/test_tokenizer_parity.c)
    target_include_directories(test_tokenizer_parity PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_tokenizer_parity PRIVATE voxcpm)
    add_test(NAME tokenizer_parity COMMAND test_tokenizer_parity voxcpm2_v2_full.gguf)

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

    # CFM/DiT parity test (requires GGUF model + Python fixtures)
    add_executable(test_cfm_parity tests/test_cfm_parity.c)
    target_include_directories(test_cfm_parity PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_cfm_parity PRIVATE voxcpm)
    add_test(NAME cfm_parity COMMAND test_cfm_parity voxcpm2_v2_full.gguf fixtures/ref)

    add_executable(test_model_loader_tensors tests/test_model_loader_tensors.c)
    target_include_directories(test_model_loader_tensors PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_model_loader_tensors PRIVATE voxcpm)
    add_test(NAME model_loader_tensors COMMAND test_model_loader_tensors)

    add_executable(test_vae_dump tools/test_vae_dump.c)
    target_include_directories(test_vae_dump PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_vae_dump PRIVATE voxcpm)

    add_executable(test_vae_only tools/test_vae_only.c)
    target_include_directories(test_vae_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_vae_only PRIVATE voxcpm)
    add_executable(test_vae_reference tools/test_vae_reference.c)
    target_include_directories(test_vae_reference PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_vae_reference PRIVATE voxcpm)
    add_executable(test_conv_layout tools/test_conv_layout.c)
    target_include_directories(test_conv_layout PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_conv_layout PRIVATE ggml ggml-base ggml-cpu)
    add_executable(test_model_tts_smoke tests/test_model_tts_smoke.c)
    target_include_directories(test_model_tts_smoke PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_model_tts_smoke PRIVATE voxcpm)
    set(VCPM_MODEL_FIXTURE "$ENV{VCPM_MODEL}")
    if(VCPM_MODEL_FIXTURE)
        add_test(NAME vae_only COMMAND test_vae_only "${VCPM_MODEL_FIXTURE}")
        set_tests_properties(vae_only PROPERTIES LABELS model)
        add_test(NAME model_tts_smoke COMMAND test_model_tts_smoke "${VCPM_MODEL_FIXTURE}")
        set_tests_properties(model_tts_smoke PROPERTIES LABELS model)
    else()
        message(STATUS "Skipping model fixture ctest registration; set VCPM_MODEL to enable model tests")
    endif()

    add_executable(test_depthwise_only tools/test_depthwise_only.c)
    target_include_directories(test_depthwise_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_depthwise_only PRIVATE voxcpm)

    add_executable(test_conv1d_only tools/test_conv1d_only.c)
    target_include_directories(test_conv1d_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_conv1d_only PRIVATE voxcpm)

    add_executable(test_model9_only tools/test_model9_only.c)
    target_include_directories(test_model9_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_model9_only PRIVATE voxcpm)

    add_executable(dump_tensors tools/dump_tensors.c)
    target_include_directories(dump_tensors PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(dump_tensors PRIVATE voxcpm)

    add_executable(check_alpha tools/check_alpha.c)
    target_include_directories(check_alpha PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(check_alpha PRIVATE voxcpm)

    add_executable(test_conv_verify tools/test_conv_verify.c)
    target_include_directories(test_conv_verify PRIVATE ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_conv_verify PRIVATE ggml)
endif()
