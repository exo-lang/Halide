#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "HalideBuffer.h"
#include "HalideRuntime.h"

#include "exo_unsharp/unsharp.h"
#include "unsharp.h"
#include "unsharp_auto_schedule.h"
#include "unsharp_v2.h"

#include "halide_benchmark.h"
#include "halide_image_io.h"

using namespace Halide::Tools;

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s in out\n", argv[0]);
        return 1;
    }

    Halide::Runtime::Buffer<float, 3> input = load_and_convert_image(argv[1]);
    Halide::Runtime::Buffer<float, 3> output(input.width(), input.height(), 3);

    double best_manual = benchmark([&]() {
        unsharp(input, output);
        output.device_sync();
    });
    // printf("Manually-tuned time: %gms\n", best_manual * 1e3);

    double best_auto = benchmark([&]() {
        unsharp_auto_schedule(input, output);
        output.device_sync();
    });
    // printf("Auto-scheduled time: %gms\n", best_auto * 1e3);

    /*
    The following code assumes the image comes with the repeated_edge
    boundary condition per-processed in order to allow for a fair 
    comparison between Exo and Halide.
    */
    Halide::Runtime::Buffer<float, 3> input_v2(input.width() + 6, input.height() + 6, 3);
    Halide::Runtime::Buffer<float, 3> output_v2(input.width(), input.height(), 3);
    std::vector<float> exo_input((input.width() + 6) * (input.height() + 6) * 3);
    std::vector<float> exo_output(input.width() * input.height() * 3);

    for (int y = 0; y < input.height() + 6; y++) {
        for (int x = 0; x < input.width() + 6; x++) {
            for (int c = 0; c < 3; c++) {
                // Repeat edge
                int new_x = std::min(std::max(x - 3, 0), input.width() - 1);
                int new_y = std::min(std::max(y - 3, 0), input.height() - 1);
                float val = input(new_x, new_y, c);
                input_v2(x, y, c) = val;
                exo_input[c * (input.width() + 6) * (input.height() + 6) + y * (input.width() + 6) + x] = val;
            }
        }
    }

    double best_manual_v2 = benchmark([&]() {
        unsharp_v2(input_v2, output_v2);
        output.device_sync();
    });
    printf("Manually-tuned v2 time: %gms\n", best_manual_v2 * 1e3);

    double best_exo = benchmark([&]() {
        // TODO: Is it a fair comparison to copy the data first for Exo? What if handling
        // the repeat_edge boundary condition is nontrivial work?
        // exo_unsharp_base(nullptr, input.width(), input.height(), &exo_output[0], &exo_input[0]);
        // exo_unsharp(nullptr, input.width(), input.height(), &exo_output[0], &exo_input[0]);
        exo_unsharp_vectorized(nullptr, input.width(), input.height(), &exo_output[0], &exo_input[0]);
    });
    printf("Exo time: %gms\n", best_exo * 1e3);

    printf("Dimensions: %d %d\n", input.width(), input.height());

    for (int y = 0; y < output_v2.width(); y++) {
        for (int x = 0; x < output_v2.height(); x++) {
            for (int c = 0; c < 3; c++) {
                if (std::abs(output(x, y, c) - output_v2(x, y, c)) > 1e-6) {
                    printf("output_v2 difference at (%d,%d, %d): %f %f\n", x, y, c, output(x, y, c), output_v2(x, y, c));
                    abort();
                }
            }
        }
    }

    for (int y = 0; y < input.width(); y++) {
        for (int x = 0; x < input.height(); x++) {
            for (int c = 0; c < 3; c++) {
                float tmp = exo_output[c * input.width() * input.height() + y * input.width() + x];
                if (std::abs(output(x, y, c) - tmp) > 1e-6) {
                    printf("exo_output difference at (%d,%d, %d): %f %f\n", x, y, c, output(x, y, c), tmp);
                    abort();
                }
            }
        }
    }

    convert_and_save_image(output, argv[2]);

    printf("Success!\n");
    return 0;
}
