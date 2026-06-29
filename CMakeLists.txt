cmake_minimum_required(VERSION 3.16)
project(voxcpm_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

# ---- ccache support ----
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found - consider installing it for faster rebuilds")
endif()

# ---- clang-tidy support (optional, not enabled by default) ----
option(VCPM_ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)
if(VCPM_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_PROGRAM clang-tidy)
    if(CLANG_TIDY_PROGRAM)
        set(CMAKE_C_CLANG_TIDY "${CLANG_TIDY_PROGRAM}")
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_PROGRAM}")
    else()
        message(WARNING "clang-tidy requested but not found")
    endif()
endif()

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
    src/audio_vae_v2_encoder.c
    src/audio_vae_v2_decoder.c
    src/gen_init.c
    src/gen_prompt.c
    src/gen_step.c
    src/gen_stop.c
    src/gen_run.c
    src/log.c
)

target_include_directories(voxcpm
    PUBLIC  include
    PRIVATE ${ggml_SOURCE_DIR}/include
)

target_link_libraries(voxcpm
    PUBLIC  ggml
)

# ---- Install rules ----
include(GNUInstallDirs)
install(TARGETS voxcpm
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
set_target_properties(voxcpm PROPERTIES PUBLIC_HEADER "include/voxcpm.h")
# ---- CPack packaging ----
set(CPACK_PACKAGE_NAME "voxcpm-c")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VoxCPM2 TTS inference engine in C/ggml")
set(CPACK_PACKAGE_VENDOR "VoxCPM-C Authors")
set(CPACK_PACKAGE_CONTACT "https://github.com/stevenke1981/voxcpm-cpp")
set(CPACK_GENERATOR "ZIP;TGZ")
if(WIN32)
    list(APPEND CPACK_GENERATOR "NSIS")
endif()
if(APPLE)
    list(APPEND CPACK_GENERATOR "DragNDrop")
endif()
include(CPack)

# ---- CLI ----
add_executable(voxcpm-c src/main.c)
target_link_libraries(voxcpm-c PRIVATE voxcpm)
install(TARGETS voxcpm-c RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---- Quantize tool (always built, no model weights needed) ----
add_executable(quantize tools/quantize.c)
target_include_directories(quantize PRIVATE ${ggml_SOURCE_DIR}/include)
target_link_libraries(quantize PRIVATE ggml)
install(TARGETS quantize RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---- Tests ----
if(VCPM_BUILD_TESTS)
    enable_testing()

    # Helper: private includes for tests
    set(VCPM_TEST_PRIVATE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # ---- Unit tests (no model weights needed) ----
    add_executable(test_smoke tests/test_smoke.c)
    target_include_directories(test_smoke PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_smoke PRIVATE voxcpm)
    add_test(NAME smoke COMMAND test_smoke)
    set_tests_properties(smoke PROPERTIES LABELS "unit")

    add_executable(test_wav tests/test_wav.c)
    target_include_directories(test_wav PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_wav PRIVATE voxcpm)
    add_test(NAME wav COMMAND test_wav)
    set_tests_properties(wav PROPERTIES LABELS "unit")

    add_executable(test_wav_writer tests/audio/test_wav_writer.c)
    target_include_directories(test_wav_writer PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_wav_writer PRIVATE voxcpm)
    add_test(NAME wav_writer COMMAND test_wav_writer)
    set_tests_properties(wav_writer PROPERTIES LABELS "unit")

    add_executable(test_sequence tests/test_sequence.c)
    target_include_directories(test_sequence PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_sequence PRIVATE voxcpm)
    add_test(NAME sequence COMMAND test_sequence)
    set_tests_properties(sequence PROPERTIES LABELS "unit")

    add_executable(test_clone_sequence tests/test_clone_sequence.c)
    target_include_directories(test_clone_sequence PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_clone_sequence PRIVATE voxcpm)
    add_test(NAME clone_sequence COMMAND test_clone_sequence)
    set_tests_properties(clone_sequence PROPERTIES LABELS "unit")

    add_executable(test_stop tests/test_stop.c)
    target_include_directories(test_stop PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_stop PRIVATE voxcpm)
    add_test(NAME stop COMMAND test_stop)
    set_tests_properties(stop PROPERTIES LABELS "unit")

    add_executable(test_cfm_schedule tests/test_cfm_schedule.c)
    target_include_directories(test_cfm_schedule PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_cfm_schedule PRIVATE voxcpm)
    add_test(NAME cfm_schedule COMMAND test_cfm_schedule)
    set_tests_properties(cfm_schedule PROPERTIES LABELS "unit")

    add_executable(test_vae_layout tests/test_vae_layout.c)
    target_include_directories(test_vae_layout PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_vae_layout PRIVATE voxcpm)
    add_test(NAME vae_layout COMMAND test_vae_layout)
    set_tests_properties(vae_layout PROPERTIES LABELS "unit")

    add_executable(test_minicpm4 tests/test_minicpm4.c)
    target_include_directories(test_minicpm4 PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_minicpm4 PRIVATE voxcpm)
    add_test(NAME minicpm4 COMMAND test_minicpm4)
    set_tests_properties(minicpm4 PROPERTIES LABELS "unit")

    add_executable(test_phase5 tests/test_phase5.c)
    target_include_directories(test_phase5 PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_phase5 PRIVATE voxcpm)
    add_test(NAME phase5 COMMAND test_phase5)
    set_tests_properties(phase5 PROPERTIES LABELS "unit")

    add_executable(test_model_loader_tensors tests/test_model_loader_tensors.c)
    target_include_directories(test_model_loader_tensors PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_model_loader_tensors PRIVATE voxcpm)
    add_test(NAME model_loader_tensors COMMAND test_model_loader_tensors)
    set_tests_properties(model_loader_tensors PROPERTIES LABELS "unit")

    # ---- Integration tests (require GGUF model file) ----
    set(VCPM_MODEL_FIXTURE "$ENV{VCPM_MODEL}")
    add_executable(test_tokenizer_parity tests/test_tokenizer_parity.c)
    target_include_directories(test_tokenizer_parity PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_tokenizer_parity PRIVATE voxcpm)
    if(VCPM_MODEL_FIXTURE)
        add_test(NAME tokenizer_parity COMMAND test_tokenizer_parity "${VCPM_MODEL_FIXTURE}")
        set_tests_properties(tokenizer_parity PROPERTIES LABELS "model")
    endif()

    # CFM/DiT parity test (requires GGUF model + Python fixtures)
    add_executable(test_cfm_parity tests/test_cfm_parity.c)
    target_include_directories(test_cfm_parity PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_cfm_parity PRIVATE voxcpm)
    if(VCPM_MODEL_FIXTURE)
        add_test(NAME cfm_parity COMMAND test_cfm_parity
                 "${VCPM_MODEL_FIXTURE}" "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/ref")
        set_tests_properties(cfm_parity PROPERTIES LABELS "model")
        add_test(NAME cfm_ar2_d8_parity COMMAND test_cfm_parity
                 "${VCPM_MODEL_FIXTURE}" "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/ref" 2 8)
        set_tests_properties(cfm_ar2_d8_parity PROPERTIES LABELS "model")
    endif()

    add_executable(test_vae_dump tests/test_vae_dump.c)
    target_include_directories(test_vae_dump PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_vae_dump PRIVATE voxcpm)

    add_executable(test_vae_only tests/test_vae_only.c)
    target_include_directories(test_vae_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_vae_only PRIVATE voxcpm)
    add_executable(test_vae_reference tests/test_vae_reference.c)
    target_include_directories(test_vae_reference PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_vae_reference PRIVATE voxcpm)
    add_executable(test_conv_layout tests/test_conv_layout.c)
    target_include_directories(test_conv_layout PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_conv_layout PRIVATE ggml ggml-base ggml-cpu)
    add_executable(test_model_tts_smoke tests/test_model_tts_smoke.c)
    target_include_directories(test_model_tts_smoke PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_model_tts_smoke PRIVATE voxcpm)
    add_executable(test_model_clone_smoke tests/test_model_clone_smoke.c)
    target_include_directories(test_model_clone_smoke PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_model_clone_smoke PRIVATE voxcpm)
    add_executable(test_vae_encoder_parity tests/test_vae_encoder_parity.c)
    target_include_directories(test_vae_encoder_parity PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_vae_encoder_parity PRIVATE voxcpm)
    add_executable(test_latent_parity tests/test_latent_parity.c)
    target_include_directories(test_latent_parity PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES})
    target_link_libraries(test_latent_parity PRIVATE voxcpm)
    add_executable(test_prompt_cuda_probe tests/test_prompt_cuda_probe.c)
    target_include_directories(test_prompt_cuda_probe PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_prompt_cuda_probe PRIVATE voxcpm)
    add_executable(test_cuda_ops tests/test_cuda_ops.c)
    target_include_directories(test_cuda_ops PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_cuda_ops PRIVATE voxcpm)
    add_executable(test_fixture_parity tests/test_fixture_parity.c)
    target_include_directories(test_fixture_parity PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_compile_definitions(test_fixture_parity PRIVATE
        FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_fixture_parity PRIVATE voxcpm)

    add_executable(debug_attention_dump tests/debug_attention_dump.c)
    target_include_directories(debug_attention_dump PRIVATE
        ${VCPM_TEST_PRIVATE_INCLUDES}
        ${ggml_SOURCE_DIR}/include)
    target_compile_definitions(debug_attention_dump PRIVATE
        FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(debug_attention_dump PRIVATE voxcpm)
    if(VCPM_MODEL_FIXTURE)
        add_test(NAME vae_only COMMAND test_vae_only "${VCPM_MODEL_FIXTURE}")
        set_tests_properties(vae_only PROPERTIES LABELS model)
        add_test(NAME model_tts_smoke COMMAND test_model_tts_smoke "${VCPM_MODEL_FIXTURE}")
        set_tests_properties(model_tts_smoke PROPERTIES LABELS model)
        add_test(NAME model_clone_smoke COMMAND test_model_clone_smoke "${VCPM_MODEL_FIXTURE}")
        set_tests_properties(model_clone_smoke PROPERTIES LABELS model)
        add_test(NAME vae_encoder_parity COMMAND test_vae_encoder_parity
                 "${VCPM_MODEL_FIXTURE}" "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/ref")
        set_tests_properties(vae_encoder_parity PROPERTIES LABELS model)
        if(VCPM_ENABLE_CUDA)
            add_test(NAME prompt_cuda_parity COMMAND test_prompt_cuda_probe
                     "${VCPM_MODEL_FIXTURE}" "Hello world.")
            set_tests_properties(prompt_cuda_parity PROPERTIES LABELS "model;cuda")
        endif()
    else()
        message(STATUS "Skipping model fixture ctest registration; set VCPM_MODEL to enable model tests")
    endif()

    add_executable(test_depthwise_only tests/test_depthwise_only.c)
    target_include_directories(test_depthwise_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_depthwise_only PRIVATE voxcpm)

    add_executable(test_conv1d_only tests/test_conv1d_only.c)
    target_include_directories(test_conv1d_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_conv1d_only PRIVATE voxcpm)

    add_executable(test_model9_only tests/test_model9_only.c)
    target_include_directories(test_model9_only PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_model9_only PRIVATE voxcpm)

    add_executable(dump_tensors tools/dump_tensors.c)
    target_include_directories(dump_tensors PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(dump_tensors PRIVATE voxcpm)

    add_executable(check_alpha tools/check_alpha.c)
    target_include_directories(check_alpha PRIVATE ${VCPM_TEST_PRIVATE_INCLUDES} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(check_alpha PRIVATE voxcpm)

    add_executable(test_conv_verify tests/test_conv_verify.c)
    target_include_directories(test_conv_verify PRIVATE ${ggml_SOURCE_DIR}/include)
    target_link_libraries(test_conv_verify PRIVATE ggml)
endif()
