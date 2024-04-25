#include <cmath>
#include <cstdint>
#include <cstdio>
#ifdef __SSE2__
#include <immintrin.h>
#elif __ARM_NEON
#include <arm_neon.h>
#endif

#include "exo_blur/blur.h"
#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

double t;

Buffer<uint16_t, 2> blur(Buffer<uint16_t, 2> in) {
    Buffer<uint16_t, 2> tmp(in.width() - 2, in.height());
    Buffer<uint16_t, 2> out(in.width() - 2, in.height() - 2);

    t = benchmark(10, 1, [&]() {
        for (int y = 0; y < tmp.height(); y++)
            for (int x = 0; x < tmp.width(); x++)
                tmp(x, y) = (in(x, y) + in(x + 1, y) + in(x + 2, y)) / 3;

        for (int y = 0; y < out.height(); y++)
            for (int x = 0; x < out.width(); x++)
                out(x, y) = (tmp(x, y) + tmp(x, y + 1) + tmp(x, y + 2)) / 3;
    });

    return out;
}

// using mm256i             0.0014 -> 0.0010
// tile size: 128->25       0.0010 -> 0.000819
// actual: ~0.000787
Buffer<uint16_t, 2> blur_fast(Buffer<uint16_t, 2> in) {
    Buffer<uint16_t, 2> out(in.width() - 2, in.height() - 2);

    t = benchmark(10, 1, [&]() {
        // __m256i one_third = _mm256_set1_epi16(21846);
        __m256i two_third = _mm256_set1_epi16(43691);
#pragma omp parallel for
        for (int yTile = 0; yTile < out.height(); yTile += 32) {
            __m256i tmp[(256 / 16) * (32 + 2)];
            for (int xTile = 0; xTile < out.width(); xTile += 256) {
                __m256i *tmpPtr = tmp;
                for (int y = 0; y < 32 + 2; y++) {
                    const uint16_t *inPtr = &(in(xTile, yTile + y));
                    for (int x = 0; x < 256; x += 16) {
                        __m256i a = _mm256_load_si256((const __m256i *)(inPtr));
                        __m256i b = _mm256_loadu_si256((const __m256i *)(inPtr + 1));
                        __m256i c = _mm256_loadu_si256((const __m256i *)(inPtr + 2));
                        __m256i sum = _mm256_add_epi16(_mm256_add_epi16(a, b), c);
                        // __m256i avg = _mm256_mulhi_epu16(sum, one_third);
                        __m256i avg = _mm256_mulhi_epu16(sum, two_third);
                        avg = _mm256_srli_epi16(avg, 1);
                        _mm256_store_si256(tmpPtr++, avg);
                        inPtr += 16;
                    }
                }
                tmpPtr = tmp;
                for (int y = 0; y < 32; y++) {
                    __m256i *outPtr = (__m256i *)(&(out(xTile, yTile + y)));
                    for (int x = 0; x < 256; x += 16) {
                        __m256i a = _mm256_load_si256(tmpPtr + (2 * 256) / 16);
                        __m256i b = _mm256_load_si256(tmpPtr + 256 / 16);
                        __m256i c = _mm256_load_si256(tmpPtr++);
                        __m256i sum = _mm256_add_epi16(_mm256_add_epi16(a, b), c);
                        // __m256i avg = _mm256_mulhi_epu16(sum, one_third);
                        __m256i avg = _mm256_mulhi_epu16(sum, two_third);
                        avg = _mm256_srli_epi16(avg, 1);
                        _mm256_store_si256(outPtr++, avg);
                    }
                }
            }
        }
    });

    return out;
}

#include "halide_blur.h"

Buffer<uint16_t, 2> blur_halide(Buffer<uint16_t, 2> in) {
    Buffer<uint16_t, 2> out(in.width() - 2, in.height() - 2);

    // Call it once to initialize the halide runtime stuff
    halide_blur(in, out);
    // Copy-out result if it's device buffer and dirty.
    out.copy_to_host();

    t = benchmark([&]() {
        // Compute the same region of the output as blur_fast (i.e., we're
        // still being sloppy with boundary conditions)
        halide_blur(in, out);
        // Sync device execution if any.
        out.device_sync();
    });

    out.copy_to_host();

    return out;
}


Buffer<uint16_t, 2> blur_exo(Buffer<uint16_t, 2> in) {
    size_t W = in.width() - 2;
    size_t H = in.height() - 2;
    
    Buffer<uint16_t, 2> out(W, H);
    t = benchmark([&]() {
        exo_blur(nullptr, W, H, out.begin(), in.begin());
    });

    return out;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Must provide a width and height argument.\n");
        return 1;
    }

    const int width = 2 + std::stoi(argv[1]);
    const int height = 2 + std::stoi(argv[2]);
    assert (width % 256 == 2 && height % 32 == 2);
    
    Buffer<uint16_t, 2> input(width, height);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand() & 0xfff;
        }
    }

    Buffer<uint16_t, 2> halide = blur_halide(input);
    double halide_time = t;

    Buffer<uint16_t, 2> exo = blur_exo(input);
    double exo_time = t;

    printf("times: %f %f\n", halide_time, exo_time);

    for (int y = 0; y < input.height() - 2; y++) {
        for (int x = 0; x < input.width() - 2; x++) {
            if (halide(x, y) != exo(x, y)) {
                printf("difference at (%d,%d): %d %d\n", x, y, halide(x, y), exo(x, y));
                abort();
            }
        }
    }

    printf("Success!\n");
}
