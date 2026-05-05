cmake_minimum_required(VERSION 3.16)
project(PI_NSGA2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimización: Release para benchmarks, Debug con sanitizers
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -fsanitize=address,undefined")

# ─── Includes ─────────────────────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ─── Fuentes ──────────────────────────────────────────────────────────────────
set(SOURCES
    src/tree_node.cpp
    src/pde_problems.cpp
    src/tsoulos_ge.cpp
    src/pi_solver.cpp
    main.cpp
)

# ─── Ejecutable principal ─────────────────────────────────────────────────────
add_executable(pi_nsga2 ${SOURCES})

# Linkar librerías matemáticas y filesystem
target_link_libraries(pi_nsga2 m stdc++fs)

# ─── Información de compilación ───────────────────────────────────────────────
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags  : ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
