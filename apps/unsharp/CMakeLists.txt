cmake_minimum_required(VERSION 3.22)
project(unsharp)

enable_testing()

add_compile_options(-march=native -O3 -fopenmp)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)
find_package(OpenMP)

# Generator
add_halide_generator(unsharp.generator SOURCES unsharp_generator.cpp)

# Filters
add_halide_library(unsharp FROM unsharp.generator)
add_halide_library(unsharp_auto_schedule FROM unsharp.generator
                   GENERATOR unsharp
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Add exo_unsharp/ to the include path
set(files "exo_unsharp/unsharp.c" "exo_unsharp/unsharp.h")
#set(files "exo_unsharp/unsharp_circular_buffer.c" "exo_unsharp/unsharp.h")
add_library(exo_unsharp ${files})

# Main executable
add_executable(unsharp_filter filter.cpp)
target_link_libraries(unsharp_filter
                      PRIVATE
                      Halide::ImageIO
                      unsharp
                      exo_unsharp
                      unsharp_auto_schedule
                      $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>)

# Test that the app actually works!
set(IMAGE ${CMAKE_CURRENT_LIST_DIR}/../images/rgba.png)
if (EXISTS ${IMAGE})
    configure_file(${IMAGE} rgba.png COPYONLY)
    add_test(NAME unsharp_filter COMMAND unsharp_filter rgba.png out.png)
    set_tests_properties(unsharp_filter PROPERTIES
                         LABELS unsharp
                         PASS_REGULAR_EXPRESSION "Success!"
                         SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
endif ()
