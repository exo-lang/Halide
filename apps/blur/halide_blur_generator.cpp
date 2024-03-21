#include "Halide.h"

namespace {

enum class BlurGPUSchedule {
    Inline,          // Fully inlining schedule.
    Cache,           // Schedule caching intermedia result of blur_x.
    Slide,           // Schedule enabling sliding window opt within each
                     // work-item or cuda thread.
    SlideVectorize,  // The same as above plus vectorization per work-item.
};

std::map<std::string, BlurGPUSchedule> blurGPUScheduleEnumMap() {
    return {
        {"inline", BlurGPUSchedule::Inline},
        {"cache", BlurGPUSchedule::Cache},
        {"slide", BlurGPUSchedule::Slide},
        {"slide_vector", BlurGPUSchedule::SlideVectorize},
    };
};

class HalideBlur : public Halide::Generator<HalideBlur> {
public:
    GeneratorParam<BlurGPUSchedule> schedule{
        "schedule",
        BlurGPUSchedule::SlideVectorize,
        blurGPUScheduleEnumMap()};
    // GeneratorParam<int> tile_x{"tile_x", 32};  // X tile.
    // GeneratorParam<int> tile_y{"tile_y", 8};   // Y tile.

    Input<Buffer<uint16_t, 2>> input{"input"};
    Output<Buffer<uint16_t, 2>> blur_y{"blur_y"};

    void generate() {
        Func blur_x("blur_x");
        Var x("x"), y("y"), xi("xi"), yi("yi");

        // The algorithm
        blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;

        // CPU schedule.
        // Compute blur_x as needed at each vector of the output.
        // Halide will store blur_x in a circular buffer so its
        // results can be re-used.
        // blur_y
        //     .split(y, y, yi, 32)
        //     .parallel(y)
        //     .vectorize(x, 16);
        // blur_x
        //     .store_at(blur_y, y)
        //     .compute_at(blur_y, x)
        //     .vectorize(x, 16);

        blur_y.tile(x, y, xi, yi, 256, 32)
                .vectorize(xi, 16).parallel(y);
        blur_x.compute_at(blur_y, x).vectorize(x, 16);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(HalideBlur, halide_blur)
