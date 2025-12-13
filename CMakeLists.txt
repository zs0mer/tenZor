cmake_minimum_required(VERSION 3.16)
project(Tenzor LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


###
# Paths
###
set(SRC_DIR   "${CMAKE_SOURCE_DIR}/Demo/src")
set(INCLUDE_DIR     "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/lib")
set(TEST_DIR  "${CMAKE_SOURCE_DIR}/Demo/test")

###
# Application executable
###
file(GLOB APP_SOURCES
    "${SRC_DIR}/*.cpp"
)

add_executable(demo ${APP_SOURCES})
target_include_directories(demo PRIVATE ${INCLUDE_DIR})

###
# Testing
###
enable_testing()

find_package(doctest REQUIRED)

file(GLOB TEST_SOURCES
    "${TEST_DIR}/*.cpp"
)

add_executable(tests ${TEST_SOURCES})

target_include_directories(tests PUBLIC ${INCLUDE_DIR})

target_link_libraries(tests
    PRIVATE
        doctest::doctest
)


add_test(
    NAME tenzor_test
    COMMAND tests --color=yes
    )


###
# Optional: sanitizers if you want them
###
option(ENABLE_ASAN "Compile with AddressSanitizer" OFF)

if(ENABLE_ASAN)
    target_compile_options(demo  PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(demo     PRIVATE -fsanitize=address)
    target_compile_options(tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(tests    PRIVATE -fsanitize=address)
endif()






