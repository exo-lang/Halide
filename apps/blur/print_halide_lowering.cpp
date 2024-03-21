#include "Halide.h"

using namespace Halide;
 

void print_blur() {
    Buffer<uint16_t, 2> input(1536, 2560);
    Var x("x"), y("y");

    // ALGORITHM
    Func blur_x("blur_x");
    blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;

    Func blur_y("blur_y");
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;

    // PRE-SCHEDULE
    Var xi("xi"), yi("yi");

    // SCHEDULE 1: does other under-the-hood optimizations
    // blur_y
    //     .split(y, y, yi, 32)
    //     .parallel(y)
    //     .vectorize(x, 16);
    // blur_x
    //     .store_at(blur_y, y)
    //     .compute_at(blur_y, x)
    //     .vectorize(x, 16);

    // SCHEDULE 2: from original Halide paper
    blur_y.tile(x, y, xi, yi, 256, 32)
            .vectorize(xi, 16).parallel(y);
    blur_x.compute_at(blur_y, x).vectorize(x, 16);

    blur_y.print_loop_nest();
    blur_y.compile_to_lowered_stmt("halide_blur.html", {}, HTML);
    // blur_y.compile_to_lowered_stmt("halide_blur.txt", {}, Text);
}


void print_unsharp() {
    Buffer<float, 3> input(1536, 2560, 3);
    Var x("x"), y("y"), c("c");

    Func kernel("kernel");
    const float kPi = 3.14159265358979310000f;
    const float sigma = 1.5;
    kernel(x) = exp(-x * x / (2 * sigma * sigma)) / (sqrtf(2 * kPi) * sigma);

    // ALGORITHM
    Func input_bounded = Halide::BoundaryConditions::repeat_edge(input);

    Func gray("gray");
    gray(x, y) = (0.299f * input_bounded(x, y, 0) +
                    0.587f * input_bounded(x, y, 1) +
                    0.114f * input_bounded(x, y, 2));

    Func blur_y("blur_y");
    blur_y(x, y) = (kernel(0) * gray(x, y) +
                    kernel(1) * (gray(x, y - 1) +
                                    gray(x, y + 1)) +
                    kernel(2) * (gray(x, y - 2) +
                                    gray(x, y + 2)) +
                    kernel(3) * (gray(x, y - 3) +
                                    gray(x, y + 3)));

    Func blur_x("blur_x");
    blur_x(x, y) = (kernel(0) * blur_y(x, y) +
                    kernel(1) * (blur_y(x - 1, y) +
                                    blur_y(x + 1, y)) +
                    kernel(2) * (blur_y(x - 2, y) +
                                    blur_y(x + 2, y)) +
                    kernel(3) * (blur_y(x - 3, y) +
                                    blur_y(x + 3, y)));

    Func sharpen("sharpen");
    sharpen(x, y) = 2 * gray(x, y) - blur_x(x, y);

    Func ratio("ratio");
    ratio(x, y) = sharpen(x, y) / gray(x, y);

    Func output("output");
    output(x, y, c) = ratio(x, y) * input(x, y, c);

    // PRE-SCHEDULE
    Var yo, yi;
    const int vec = 8; // 256 / 32 for AVX2

    // SCHEDULE
    output.split(y, yo, yi, 32)
        .vectorize(x, vec)
        .parallel(yo)
        .reorder(x, c, yi, yo);
    gray.compute_at(output, yi)
        .store_at(output, yo)
        .vectorize(x, vec);
    blur_y.compute_at(output, yi)
        .store_at(output, yo)
        .vectorize(x, vec);
    ratio.compute_at(output, yi)
        .store_at(output, yo)
        .vectorize(x, vec);

    output.print_loop_nest();
    output.compile_to_lowered_stmt("halide_unsharp.html", {}, HTML);
}


int main(int argc, char **argv) {
    // print_blur();
    print_unsharp();
}