cmake_minimum_required(VERSION 3.24)
project(test_gemm LANGUAGES CXX CUDA)

# CUTLASS root: test_gemm 位于 ${CUTLASS_DIR}/test_gemm
set(CUTLASS_DIR "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Path to CUTLASS source directory")

# Hopper: sm_90a
set(CUDA_ARCH "90a" CACHE STRING "CUDA architecture (e.g. 90 or 90a)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# --- build target ---
add_executable(run_gemm
  run_sm90_bf16_tma_coop_gemm.cu
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(run_gemm PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
)

# 关键：把头文件“挂上去”，方便依赖跟踪/修改触发重编译（不是单独编译）
target_sources(run_gemm PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/utils.cuh
  ${CMAKE_CURRENT_LIST_DIR}/gemm.cuh
)

target_include_directories(run_gemm PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}            # for "utils.cuh" / "gemm.cuh"
  ${CUTLASS_DIR}/include               # cutlass/*
  ${CUTLASS_DIR}/tools/util/include    # cutlass/util/*
  ${CUTLASS_DIR}/examples/common       # helper.h (if used)
)

set_target_properties(run_gemm PROPERTIES
  CUDA_ARCHITECTURES ${CUDA_ARCH}
  CUDA_STANDARD 17
  CXX_STANDARD 17
)

target_compile_options(run_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -lineinfo
    -O3
  >
  $<$<COMPILE_LANGUAGE:CXX>:
    -O3
  >
)

# 可选：少点模板 warning
target_compile_options(run_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>
)

message(STATUS "CUTLASS_DIR = ${CUTLASS_DIR}")
message(STATUS "CUDA_ARCH   = ${CUDA_ARCH}")
