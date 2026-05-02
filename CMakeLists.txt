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

# # OpenMP
find_package(OpenMP REQUIRED)

# # Doctest
find_package(doctest REQUIRED)

# # Debugging flags
#set(CMAKE_CXX_FLAGS_DEBUG
#    "-O0 -g3 -fno-inline -fno-omit-frame-pointer -fno-optimize-sibling-calls"
#)

# # Sanitizers

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -g")
# set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")

# # Paths
set(SRC_DIR "${CMAKE_SOURCE_DIR}/Demo/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/lib/headers")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/Demo/test")

set(CUDA_SRC_DIR "${CMAKE_SOURCE_DIR}/lib/cuda")
set(CUDA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/headers")

# # CUDA Library
add_library(tenzor_cuda
    "${CUDA_SRC_DIR}/kernels.cu"
)

target_include_directories(tenzor_cuda PRIVATE ${CUDA_INCLUDE_DIR})
target_link_libraries(tenzor_cuda PRIVATE CUDA::cudart CUDA::cublas)

# # Executable
add_executable(demo "${SRC_DIR}/demo.cpp")
target_include_directories(demo PRIVATE ${INCLUDE_DIR})
target_compile_options(demo PRIVATE
    -O3
    -march=x86-64-v3
    -ffast-math
)
target_link_libraries(demo PRIVATE tenzor_cuda OpenMP::OpenMP_CXX)

# # Testing =====================================================================
add_executable(Tallocator "${TEST_DIR}/allocator.cpp")

target_include_directories(Tallocator PUBLIC ${INCLUDE_DIR})

target_link_libraries(Tallocator
    PRIVATE
    doctest::doctest
    tenzor_cuda
    OpenMP::OpenMP_CXX
)

# # buffer ----------------------------
add_executable(Tbuffer "${TEST_DIR}/buffer.cpp")

target_include_directories(Tbuffer PUBLIC ${INCLUDE_DIR})

target_link_libraries(Tbuffer
    PRIVATE
    doctest::doctest
    tenzor_cuda
    OpenMP::OpenMP_CXX
)

# # tensor ----------------------------
add_executable(Ttensor "${TEST_DIR}/tensor.cpp")

target_include_directories(Ttensor PUBLIC ${INCLUDE_DIR})

target_link_libraries(Ttensor
    PRIVATE
    doctest::doctest
    tenzor_cuda
    OpenMP::OpenMP_CXX
)

# # math classes ----------------------------
add_executable(Tother_classes "${TEST_DIR}/other_classes.cpp")

target_include_directories(Tother_classes PUBLIC ${INCLUDE_DIR})

target_link_libraries(Tother_classes
    PRIVATE
    doctest::doctest
    tenzor_cuda
    OpenMP::OpenMP_CXX
)

# # operations ----------------------------
add_executable(Toperations "${TEST_DIR}/operations.cpp")

target_include_directories(Toperations PUBLIC ${INCLUDE_DIR})

target_link_libraries(Toperations
    PRIVATE
    doctest::doctest
    tenzor_cuda
    OpenMP::OpenMP_CXX
)

# # # gpu function ----------------------------
add_executable(Tgpu_function "${TEST_DIR}/gpu_function.cpp")

target_include_directories(Tgpu_function PUBLIC ${INCLUDE_DIR})

target_link_libraries(Tgpu_function
    PRIVATE
    doctest::doctest
    tenzor_cuda
    OpenMP::OpenMP_CXX
)

# # ---------------------------------
enable_testing()

add_test(
    NAME allocator
    COMMAND Tallocator --color=yes
)

add_test(
    NAME buffer
    COMMAND Tbuffer --color=yes
)

add_test(
    NAME tensor
    COMMAND Ttensor --color=yes
)

add_test(
    NAME other_classes
    COMMAND Tother_classes --color=yes
)

add_test(
    NAME operations
    COMMAND Toperations --color=yes
)

add_test(
    NAME gpu_function
    COMMAND Tgpu_function --color=yes
)
