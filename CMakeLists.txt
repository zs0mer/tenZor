cmake_minimum_required(VERSION 3.16)

option(USE_CUDA "Enable CUDA support" ON)
option(TESTS "Enable tests" ON)

if(USE_CUDA)
  project(TenZor LANGUAGES CXX CUDA)
else()
  project(TenZor LANGUAGES CXX)
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Build type flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v3 -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3 -fno-inline -fno-omit-frame-pointer")

# # nvcc standard
if(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# # OpenMP
find_package(OpenMP REQUIRED)

# # Doctest
if(TESTS)
  find_package(doctest REQUIRED)
endif()

# # Paths
set(SRC_DIR "${CMAKE_SOURCE_DIR}/Demo/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/lib/headers")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/Demo/test")

# # CUDA Library
if(USE_CUDA)
  set(CUDA_SRC_DIR "${CMAKE_SOURCE_DIR}/lib/cuda")

  add_library(tenzor_cuda "${CUDA_SRC_DIR}/kernels.cu")
  target_include_directories(tenzor_cuda PRIVATE ${INCLUDE_DIR})
  target_link_libraries(tenzor_cuda PRIVATE CUDA::cudart CUDA::cublas)
endif()

# # Executable
add_executable(demo "${SRC_DIR}/demo.cpp")
target_include_directories(demo PRIVATE ${INCLUDE_DIR})
target_link_libraries(demo PRIVATE OpenMP::OpenMP_CXX)

if(USE_CUDA)
  target_compile_definitions(demo PRIVATE TZ_CUDA_AVAILABLE=1)
  target_link_libraries(demo PRIVATE tenzor_cuda)
else()
  target_compile_definitions(demo PRIVATE TZ_CUDA_AVAILABLE=0)
endif()

add_executable(demo2 "${SRC_DIR}/demo2.cpp")
target_include_directories(demo2 PRIVATE ${INCLUDE_DIR})
target_link_libraries(demo2 PRIVATE OpenMP::OpenMP_CXX)

if(USE_CUDA)
  target_compile_definitions(demo2 PRIVATE TZ_CUDA_AVAILABLE=1)
  target_link_libraries(demo2 PRIVATE tenzor_cuda)
else()
  target_compile_definitions(demo2 PRIVATE TZ_CUDA_AVAILABLE=0)
endif()

# # Testing - only with GPU =====================================================================
if(USE_CUDA AND TESTS)
  add_executable(Tallocator     "${TEST_DIR}/allocator.cpp")
  add_executable(Tbuffer        "${TEST_DIR}/buffer.cpp")
  add_executable(Ttensor        "${TEST_DIR}/tensor.cpp")
  add_executable(Tother_classes "${TEST_DIR}/other_classes.cpp")
  add_executable(Toperations    "${TEST_DIR}/operations.cpp")
  add_executable(Tgpu_function  "${TEST_DIR}/gpu_function.cpp")
  add_executable(Tautograd      "${TEST_DIR}/autograd.cpp")

  foreach(T Tallocator Tbuffer Ttensor Tother_classes Toperations Tgpu_function Tautograd)
    target_include_directories(${T} PRIVATE ${INCLUDE_DIR})
    target_compile_definitions(${T} PRIVATE TZ_CUDA_AVAILABLE=1)
    target_link_libraries(${T} PRIVATE doctest::doctest tenzor_cuda OpenMP::OpenMP_CXX)
  endforeach()

  # # ---------------------------------
  enable_testing()

  add_test(NAME allocator     COMMAND Tallocator     --color=yes)
  add_test(NAME buffer        COMMAND Tbuffer        --color=yes)
  add_test(NAME tensor        COMMAND Ttensor        --color=yes)
  add_test(NAME other_classes COMMAND Tother_classes --color=yes)
  add_test(NAME operations    COMMAND Toperations    --color=yes)
  add_test(NAME gpu_function  COMMAND Tgpu_function  --color=yes)
  add_test(NAME autograd      COMMAND Tautograd      --color=yes)

endif()
