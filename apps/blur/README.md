To compile a <cpp_file> to an <executable>:
```
g++ <cpp_file> -g -std=c++17 -I ~/Halide-16.0.0-x86-64-linux/include/ -L ~/Halide-16.0.0-x86-64-linux/lib/ -lHalide -lpthread -ldl -o <executable>
```

To execute an <executable>:
```
LD_LIBRARY_PATH=~/Halide-16.0.0-x86-64-linux/lib <executable>
```