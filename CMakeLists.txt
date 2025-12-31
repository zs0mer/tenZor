cmake_minimum_required(VERSION 3.16)
project(Tenzor LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g3 -fno-inline -fno-omit-frame-pointer -fno-optimize-sibling-calls"
)

###
# Sanitizers
###


#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -g")
#set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")



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



enable_testing()

add_test(
    NAME allocater_basic
    COMMAND tests --color=yes
    )


