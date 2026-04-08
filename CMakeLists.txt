cmake_minimum_required(VERSION 3.16)
project(Tenzor LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # nvcc standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

# ! THIS DEPENDS ON GPU
set(CMAKE_CUDA_ARCHITECTURES 61)

# Debugging flags
set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g3 -fno-inline -fno-omit-frame-pointer -fno-optimize-sibling-calls"
)

# # ##
# # Sanitizers
# # ##

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -g")
# set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")

# # ##
# # Paths
# # ##
set(SRC_DIR "${CMAKE_SOURCE_DIR}/Demo/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/lib/headers")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/Demo/test")

set(CUDA_SRC_DIR "${CMAKE_SOURCE_DIR}/lib/cuda")
set(CUDA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/headers")

# # ##
# # CUDA Library
# # ##
add_library(tenzor_cuda
    "${CUDA_SRC_DIR}/kernels.cu"
)

target_include_directories(tenzor_cuda PRIVATE ${CUDA_INCLUDE_DIR})
target_link_libraries(tenzor_cuda PRIVATE CUDA::cudart)

# # ##
# # Application executable
# # ##
file(GLOB APP_SOURCES
    "${SRC_DIR}/*.cpp"
)

add_executable(demo "${SRC_DIR}/demo.cpp")
target_include_directories(demo PRIVATE ${INCLUDE_DIR})
target_link_libraries(demo PRIVATE tenzor_cuda)

add_executable(bench "${SRC_DIR}/bench.cpp")
target_include_directories(bench PRIVATE ${INCLUDE_DIR})
target_compile_options(bench PRIVATE
    -O3
    -march=x86-64-v3
    -ffast-math

    # -funroll-loops
    # -fopt-info-vec
    # -fno-omit-frame-pointer
)
target_link_libraries(bench PRIVATE tenzor_cuda)

add_executable(alloc_bench "${SRC_DIR}/alloc_bench.cpp")
target_include_directories(alloc_bench PRIVATE ${INCLUDE_DIR})
target_compile_options(alloc_bench PRIVATE
    -O3
    -march=x86-64-v3
    -ffast-math

    # -funroll-loops
    # -fopt-info-vec
    # -fno-omit-frame-pointer
)
target_link_libraries(alloc_bench PRIVATE tenzor_cuda)

# # ## ===========================================================================
# # Testing =====================================================================
# # ## ===========================================================================
find_package(doctest REQUIRED)

# # allocater ------------------------
add_executable(Tallocater "${TEST_DIR}/allocater.cpp")

target_include_directories(Tallocater PUBLIC ${INCLUDE_DIR})

target_link_libraries(Tallocater
    PRIVATE
    doctest::doctest
)
target_link_libraries(Tallocater PRIVATE tenzor_cuda)

# # buffer ----------------------------
add_executable(Tbuffer "${TEST_DIR}/buffer.cpp")

target_include_directories(Tbuffer PUBLIC ${INCLUDE_DIR})

target_link_libraries(Tbuffer
    PRIVATE
    doctest::doctest
)
target_link_libraries(Tbuffer PRIVATE tenzor_cuda)

# # ---------------------------------
enable_testing()

add_test(
    NAME allocater
    COMMAND Tallocater --color=yes
)

add_test(
    NAME buffer
    COMMAND Tbuffer --color=yes
)
