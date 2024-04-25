#!/bin/bash

# Define the program to run
# program="taskset -c 1 build/blur_test"
program="build/blur_test"

# Define the set of arguments
w_args=(1280 2560 5120)
h_args=(960 1920 3840)

# Loop over the Cartesian product of arguments
for arg1 in "${w_args[@]}"; do
    for arg2 in "${h_args[@]}"; do
        echo $arg1 $arg2
        $program $arg1 $arg2
    done
done