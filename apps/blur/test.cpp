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
    Buffer<uint16_t, 2> tmp(in.width() - 8, in.height());
    Buffer<uint16_t, 2> out(in.width() - 8, in.height() - 2);

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
    Buffer<uint16_t, 2> out(in.width() - 8, in.height() - 2);

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
    Buffer<uint16_t, 2> out(in.width() - 8, in.height() - 2);

    // Call it once to initialize the halide runtime stuff
    halide_blur(in, out);
    // Copy-out result if it's device buffer and dirty.
    out.copy_to_host();

    t = benchmark(1000, 1, [&]() {
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
    size_t W = in.width() - 8;
    size_t H = in.height() - 2;
    
    // other code seems to ignore the last 6 columns of the image
    uint16_t* inp = (uint16_t*) malloc((W + 2) * (H + 2) * sizeof(uint16_t));

    // copy from Halide Buffer to array
    for (int y = 0; y < H + 2; y++) {
        for (int x = 0; x < W + 2; x++) {
            inp[y * (W + 2) + x] = in(x, y);
        }
    }

    Buffer<uint16_t, 2> out(W, H);
    uint16_t* blur_y = (uint16_t*) malloc(W * H * sizeof(uint16_t));

    // TODO: static leads to a race condition in C++
    // TODO: for some reason the code gets slower
    t = benchmark(1000, 1, [&]() {
        exo_blur(nullptr, W, H, blur_y, inp);
    });

    // copy from array to Halide Buffer
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            out(x, y) = blur_y[y * W + x];
        }
    }

    return out;
}


int main(int argc, char **argv) {
    const auto *md = halide_blur_metadata();
    const bool is_hexagon = strstr(md->target, "hvx_128") || strstr(md->target, "hvx_64");

    // The Hexagon simulator can't allocate as much memory as the above wants.
    const int width = is_hexagon ? 648 : 5120 + 8;
    const int height = is_hexagon ? 482 : 3840 + 2;

    Buffer<uint16_t, 2> input(width, height);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand() & 0xfff;
        }
    }

    Buffer<uint16_t, 2> blurry = blur(input);
    double slow_time = t;

    Buffer<uint16_t, 2> speedy = blur_fast(input);
    double fast_time = t;

    Buffer<uint16_t, 2> halide = blur_halide(input);
    double halide_time = t;

    Buffer<uint16_t, 2> exo = blur_exo(input);
    double exo_time = t;

    printf("times: %f %f %f %f\n", slow_time, fast_time, halide_time, exo_time);

    for (int y = 64; y < input.height() - 64; y++) {
        for (int x = 64; x < input.width() - 64; x++) {
            if (blurry(x, y) != speedy(x, y) || blurry(x, y) != halide(x, y) || blurry(x, y) != exo(x, y)) {
                printf("difference at (%d,%d): %d %d %d %d\n", x, y, blurry(x, y), speedy(x, y), halide(x, y), exo(x, y));
                abort();
            }
        }
    }

    printf("Success!\n");
}
