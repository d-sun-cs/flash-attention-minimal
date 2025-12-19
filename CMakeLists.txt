cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flashattn LANGUAGES CXX CUDA)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# set(CMAKE_CUDA_ARCHITECTURES "naive")   # 根据你的 GPU 架构，比如 RTX 30 系列是 86

# 指定 PyTorch 的 CMake 配置路径
# set(Torch_DIR "/opt/conda/lib/python3.11/site-packages/torch/share/cmake/Torch")

# 或者用 CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "/opt/conda/lib/python3.11/site-packages/torch/share/cmake")
# set(CMAKE_PREFIX_PATH "/root/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 根据构建类型添加编译标志
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     # C/C++ 调试标志
#     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    
#     # ⭐ CUDA 核心调试标志 ⭐
#     # -g 用于主机代码 (Host Code) 的调试信息
#     # -G 用于设备代码 (Device Code/Kernel) 的调试信息
#     # -lineinfo 增加额外的行信息支持
#     set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -lineinfo")
# endif()

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

add_executable(bench bench.cpp flash.cu)
target_link_libraries(bench "${TORCH_LIBRARIES}")
set_property(TARGET bench PROPERTY CXX_STANDARD 17)

add_executable(my_bench my_bench.cpp my_flash.cu flash.cu)
target_link_libraries(my_bench "${TORCH_LIBRARIES}")
set_property(TARGET my_bench PROPERTY CXX_STANDARD 17)

# debug cuda kernel
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(my_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)