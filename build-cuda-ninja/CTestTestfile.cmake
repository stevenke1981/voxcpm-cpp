# CMake generated Testfile for 
# Source directory: D:/voxcpm-cpp
# Build directory: D:/voxcpm-cpp/build-cuda-ninja
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(smoke "D:/voxcpm-cpp/build-cuda-ninja/test_smoke.exe")
set_tests_properties(smoke PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;149;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(wav "D:/voxcpm-cpp/build-cuda-ninja/test_wav.exe")
set_tests_properties(wav PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;155;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(wav_writer "D:/voxcpm-cpp/build-cuda-ninja/test_wav_writer.exe")
set_tests_properties(wav_writer PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;161;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(sequence "D:/voxcpm-cpp/build-cuda-ninja/test_sequence.exe")
set_tests_properties(sequence PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;167;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(minicpm4 "D:/voxcpm-cpp/build-cuda-ninja/test_minicpm4.exe")
set_tests_properties(minicpm4 PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;175;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(phase5 "D:/voxcpm-cpp/build-cuda-ninja/test_phase5.exe")
set_tests_properties(phase5 PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;183;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(model_loader_tensors "D:/voxcpm-cpp/build-cuda-ninja/test_model_loader_tensors.exe")
set_tests_properties(model_loader_tensors PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;189;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(tokenizer_parity "D:/voxcpm-cpp/build-cuda-ninja/test_tokenizer_parity.exe" "voxcpm2_v2_full.gguf")
set_tests_properties(tokenizer_parity PROPERTIES  LABELS "model" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;196;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
add_test(cfm_parity "D:/voxcpm-cpp/build-cuda-ninja/test_cfm_parity.exe" "voxcpm2_v2_full.gguf" "fixtures/ref")
set_tests_properties(cfm_parity PROPERTIES  LABELS "model" _BACKTRACE_TRIPLES "D:/voxcpm-cpp/CMakeLists.txt;205;add_test;D:/voxcpm-cpp/CMakeLists.txt;0;")
subdirs("_deps/ggml-build")
