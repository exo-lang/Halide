#include "unsharp.h"



#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>



// exo_unsharp(
//     W : size,
//     H : size,
//     output : f32[3, H, W] @DRAM,
//     input : f32[3, H + 6, W + 6] @DRAM
// )
void exo_unsharp( void *ctxt, int_fast32_t W, int_fast32_t H, float* output, const float* input ) {
EXO_ASSUME(H % 32 == 0);
float *rgb = (float*) malloc(3 * sizeof(*rgb));
rgb[0] = 0.299f;
rgb[1] = 0.587f;
rgb[2] = 0.114f;
float *kernel = (float*) malloc(4 * sizeof(*kernel));
kernel[0] = 0.2659615202676218f;
kernel[1] = 0.2129653370149015f;
kernel[2] = 0.10934004978399575f;
kernel[3] = 0.035993977675458706f;
float *two = (float*) malloc(1 * sizeof(*two));
two[0] = 2.0f;
#pragma omp parallel for
for (int_fast32_t y = 0; y < ((H) / (32)); y++) {
  float *gray = (float*) malloc(38 * (6 + W) * sizeof(*gray));
  for (int_fast32_t yi = 0; yi < 6; yi++) {
    for (int_fast32_t x = 0; x < 6 + W; x++) {
      gray[yi * (6 + W) + x] = rgb[0] * input[(yi + 32 * y) * (W + 6) + x] + rgb[1] * input[(H + 6) * (W + 6) + (yi + 32 * y) * (W + 6) + x] + rgb[2] * input[2 * (H + 6) * (W + 6) + (yi + 32 * y) * (W + 6) + x];
    }
  }
  float *ratio = (float*) malloc(1 * W * sizeof(*ratio));
  float *blur_y = (float*) malloc(1 * (6 + W) * sizeof(*blur_y));
  for (int_fast32_t yi = 0; yi < 32; yi++) {
    for (int_fast32_t x = 0; x < 6 + W; x++) {
      gray[(6 + yi) * (6 + W) + x] = rgb[0] * input[(6 + yi + 32 * y) * (W + 6) + x] + rgb[1] * input[(H + 6) * (W + 6) + (6 + yi + 32 * y) * (W + 6) + x] + rgb[2] * input[2 * (H + 6) * (W + 6) + (6 + yi + 32 * y) * (W + 6) + x];
    }
    for (int_fast32_t x = 0; x < 6 + W; x++) {
      blur_y[x] = kernel[0] * gray[(3 + yi) * (6 + W) + x] + kernel[1] * (gray[(2 + yi) * (6 + W) + x] + gray[(4 + yi) * (6 + W) + x]) + kernel[2] * (gray[(1 + yi) * (6 + W) + x] + gray[(5 + yi) * (6 + W) + x]) + kernel[3] * (gray[yi * (6 + W) + x] + gray[(6 + yi) * (6 + W) + x]);
    }
    for (int_fast32_t x = 0; x < W; x++) {
      ratio[x] = (two[0] * gray[(3 + yi) * (6 + W) + 3 + x] - (kernel[0] * blur_y[3 + x] + kernel[1] * (blur_y[2 + x] + blur_y[4 + x]) + kernel[2] * (blur_y[1 + x] + blur_y[5 + x]) + kernel[3] * (blur_y[x] + blur_y[6 + x]))) / gray[(3 + yi) * (6 + W) + 3 + x];
    }
    for (int_fast32_t c = 0; c < 3; c++) {
      for (int_fast32_t x = 0; x < W; x++) {
        output[c * H * W + (yi + 32 * y) * W + x] = ratio[x] * input[c * (H + 6) * (W + 6) + (3 + yi + 32 * y) * (W + 6) + 3 + x];
      }
    }
  }
  free(blur_y);
  free(ratio);
  free(gray);
}
free(two);
free(kernel);
free(rgb);
}

// exo_unsharp_base(
//     W : size,
//     H : size,
//     output : f32[3, H, W] @DRAM,
//     input : f32[3, H + 6, W + 6] @DRAM
// )
void exo_unsharp_base( void *ctxt, int_fast32_t W, int_fast32_t H, float* output, const float* input ) {
EXO_ASSUME(H % 32 == 0);
float *rgb = (float*) malloc(3 * sizeof(*rgb));
rgb[0] = 0.299f;
rgb[1] = 0.587f;
rgb[2] = 0.114f;
float *kernel = (float*) malloc(4 * sizeof(*kernel));
kernel[0] = 0.2659615202676218f;
kernel[1] = 0.2129653370149015f;
kernel[2] = 0.10934004978399575f;
kernel[3] = 0.035993977675458706f;
float *two = (float*) malloc(1 * sizeof(*two));
two[0] = 2.0f;
float *gray = (float*) malloc((H + 6) * (W + 6) * sizeof(*gray));
for (int_fast32_t y = 0; y < H + 6; y++) {
  for (int_fast32_t x = 0; x < W + 6; x++) {
    gray[y * (W + 6) + x] = rgb[0] * input[y * (W + 6) + x] + rgb[1] * input[(H + 6) * (W + 6) + y * (W + 6) + x] + rgb[2] * input[2 * (H + 6) * (W + 6) + y * (W + 6) + x];
  }
}
free(rgb);
float *blur_y = (float*) malloc(H * (W + 6) * sizeof(*blur_y));
for (int_fast32_t y = 0; y < H; y++) {
  for (int_fast32_t x = 0; x < W + 6; x++) {
    blur_y[y * (W + 6) + x] = kernel[0] * gray[(y + 3) * (W + 6) + x] + kernel[1] * (gray[(y + 2) * (W + 6) + x] + gray[(y + 4) * (W + 6) + x]) + kernel[2] * (gray[(y + 1) * (W + 6) + x] + gray[(y + 5) * (W + 6) + x]) + kernel[3] * (gray[y * (W + 6) + x] + gray[(y + 6) * (W + 6) + x]);
  }
}
float *blur_x = (float*) malloc(H * W * sizeof(*blur_x));
for (int_fast32_t y = 0; y < H; y++) {
  for (int_fast32_t x = 0; x < W; x++) {
    blur_x[y * W + x] = kernel[0] * blur_y[y * (W + 6) + x + 3] + kernel[1] * (blur_y[y * (W + 6) + x + 2] + blur_y[y * (W + 6) + x + 4]) + kernel[2] * (blur_y[y * (W + 6) + x + 1] + blur_y[y * (W + 6) + x + 5]) + kernel[3] * (blur_y[y * (W + 6) + x] + blur_y[y * (W + 6) + x + 6]);
  }
}
free(blur_y);
free(kernel);
float *sharpen = (float*) malloc(H * W * sizeof(*sharpen));
for (int_fast32_t y = 0; y < H; y++) {
  for (int_fast32_t x = 0; x < W; x++) {
    sharpen[y * W + x] = two[0] * gray[(y + 3) * (W + 6) + x + 3] - blur_x[y * W + x];
  }
}
free(blur_x);
free(two);
float *ratio = (float*) malloc(H * W * sizeof(*ratio));
for (int_fast32_t y = 0; y < H; y++) {
  for (int_fast32_t x = 0; x < W; x++) {
    ratio[y * W + x] = sharpen[y * W + x] / gray[(y + 3) * (W + 6) + x + 3];
  }
}
free(sharpen);
free(gray);
for (int_fast32_t y = 0; y < H; y++) {
  for (int_fast32_t c = 0; c < 3; c++) {
    for (int_fast32_t x = 0; x < W; x++) {
      output[c * H * W + y * W + x] = ratio[y * W + x] * input[c * (H + 6) * (W + 6) + (y + 3) * (W + 6) + x + 3];
    }
  }
}
free(ratio);
}

// exo_unsharp_vectorized(
//     W : size,
//     H : size,
//     output : f32[3, H, W] @DRAM,
//     input : f32[3, H + 6, W + 6] @DRAM
// )
void exo_unsharp_vectorized( void *ctxt, int_fast32_t W, int_fast32_t H, float* output, const float* input ) {
EXO_ASSUME(H % 32 == 0);
float *rgb = (float*) malloc(3 * sizeof(*rgb));
rgb[0] = 0.299f;
rgb[1] = 0.587f;
rgb[2] = 0.114f;
float *kernel = (float*) malloc(4 * sizeof(*kernel));
kernel[0] = 0.2659615202676218f;
kernel[1] = 0.2129653370149015f;
kernel[2] = 0.10934004978399575f;
kernel[3] = 0.035993977675458706f;
float *two = (float*) malloc(1 * sizeof(*two));
two[0] = 2.0f;
#pragma omp parallel for
for (int_fast32_t y = 0; y < ((H) / (32)); y++) {
  float *gray = (float*) malloc(38 * (6 + W) * sizeof(*gray));
  for (int_fast32_t yi = 0; yi < 6; yi++) {
    for (int_fast32_t xo = 0; xo < ((13 + W) / (8)) - 1; xo++) {
      __m256 var0;
      __m256 var1;
      __m256 var2;
      __m256 var3;
      __m256 var4;
      __m256 var5;
      __m256 var6;
      __m256 var7;
      __m256 var8;
      __m256 var9;
      __m256 var10;
      var3 = _mm256_broadcast_ss(&rgb[0]);
      var4 = _mm256_loadu_ps(&input[(yi + 32 * y) * (W + 6) + 8 * xo]);
      var2 = _mm256_mul_ps(var3, var4);
      var6 = _mm256_broadcast_ss(&rgb[1]);
      var7 = _mm256_loadu_ps(&input[(H + 6) * (W + 6) + (yi + 32 * y) * (W + 6) + 8 * xo]);
      var5 = _mm256_mul_ps(var6, var7);
      var1 = _mm256_add_ps(var2, var5);
      var9 = _mm256_broadcast_ss(&rgb[2]);
      var10 = _mm256_loadu_ps(&input[(2) * ((H + 6) * (W + 6)) + (yi + 32 * y) * (W + 6) + 8 * xo]);
      var8 = _mm256_mul_ps(var9, var10);
      var0 = _mm256_add_ps(var1, var8);
      _mm256_storeu_ps(&gray[(yi % 8) * (6 + W) + 8 * xo], var0);
    }
    for (int_fast32_t xo = ((13 + W) / (8)) - 1; xo < ((13 + W) / (8)); xo++) {
      __m256 var0;
      __m256 var1;
      __m256 var2;
      __m256 var3;
      __m256 var4;
      __m256 var5;
      __m256 var6;
      __m256 var7;
      __m256 var8;
      __m256 var9;
      __m256 var10;
      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&rgb[0]);
var3 = _mm256_blendv_ps (var3, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var4 = _mm256_maskload_ps(&input[(yi + 32 * y) * (W + 6) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var3, var4);
var2 = _mm256_blendv_ps (var2, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&rgb[1]);
var6 = _mm256_blendv_ps (var6, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var7 = _mm256_maskload_ps(&input[(H + 6) * (W + 6) + (yi + 32 * y) * (W + 6) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var6, var7);
var5 = _mm256_blendv_ps (var5, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var2, var5);
var1 = _mm256_blendv_ps (var1, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&rgb[2]);
var9 = _mm256_blendv_ps (var9, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var10 = _mm256_maskload_ps(&input[(2) * ((H + 6) * (W + 6)) + (yi + 32 * y) * (W + 6) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var9, var10);
var8 = _mm256_blendv_ps (var8, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var1, var8);
var0 = _mm256_blendv_ps (var0, add, _mm256_castsi256_ps(cmp));
}

      
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&gray[(yi % 8) * (6 + W) + 8 * xo], cmp, var0);
    }
    
    }
  }
  float *ratio = (float*) malloc(1 * W * sizeof(*ratio));
  float *blur_y = (float*) malloc(1 * (6 + W) * sizeof(*blur_y));
  for (int_fast32_t yi = 0; yi < 32; yi++) {
    for (int_fast32_t xo = 0; xo < ((13 + W) / (8)) - 1; xo++) {
      __m256 var11;
      __m256 var12;
      __m256 var13;
      __m256 var14;
      __m256 var15;
      __m256 var16;
      __m256 var17;
      __m256 var18;
      __m256 var19;
      __m256 var20;
      __m256 var21;
      var14 = _mm256_broadcast_ss(&rgb[0]);
      var15 = _mm256_loadu_ps(&input[(6 + yi + 32 * y) * (W + 6) + 8 * xo]);
      var13 = _mm256_mul_ps(var14, var15);
      var17 = _mm256_broadcast_ss(&rgb[1]);
      var18 = _mm256_loadu_ps(&input[(H + 6) * (W + 6) + (6 + yi + 32 * y) * (W + 6) + 8 * xo]);
      var16 = _mm256_mul_ps(var17, var18);
      var12 = _mm256_add_ps(var13, var16);
      var20 = _mm256_broadcast_ss(&rgb[2]);
      var21 = _mm256_loadu_ps(&input[(2) * ((H + 6) * (W + 6)) + (6 + yi + 32 * y) * (W + 6) + 8 * xo]);
      var19 = _mm256_mul_ps(var20, var21);
      var11 = _mm256_add_ps(var12, var19);
      _mm256_storeu_ps(&gray[((6 + yi) % 8) * (6 + W) + 8 * xo], var11);
    }
    for (int_fast32_t xo = ((13 + W) / (8)) - 1; xo < ((13 + W) / (8)); xo++) {
      __m256 var11;
      __m256 var12;
      __m256 var13;
      __m256 var14;
      __m256 var15;
      __m256 var16;
      __m256 var17;
      __m256 var18;
      __m256 var19;
      __m256 var20;
      __m256 var21;
      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&rgb[0]);
var14 = _mm256_blendv_ps (var14, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var15 = _mm256_maskload_ps(&input[(6 + yi + 32 * y) * (W + 6) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var14, var15);
var13 = _mm256_blendv_ps (var13, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&rgb[1]);
var17 = _mm256_blendv_ps (var17, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var18 = _mm256_maskload_ps(&input[(H + 6) * (W + 6) + (6 + yi + 32 * y) * (W + 6) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var17, var18);
var16 = _mm256_blendv_ps (var16, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var13, var16);
var12 = _mm256_blendv_ps (var12, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&rgb[2]);
var20 = _mm256_blendv_ps (var20, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var21 = _mm256_maskload_ps(&input[(2) * ((H + 6) * (W + 6)) + (6 + yi + 32 * y) * (W + 6) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var20, var21);
var19 = _mm256_blendv_ps (var19, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var12, var19);
var11 = _mm256_blendv_ps (var11, add, _mm256_castsi256_ps(cmp));
}

      
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&gray[((6 + yi) % 8) * (6 + W) + 8 * xo], cmp, var11);
    }
    
    }
    for (int_fast32_t xo = 0; xo < ((13 + W) / (8)) - 1; xo++) {
      __m256 var22;
      __m256 var23;
      __m256 var24;
      __m256 var25;
      __m256 var26;
      __m256 var27;
      __m256 var28;
      __m256 var29;
      __m256 var30;
      __m256 var31;
      __m256 var32;
      __m256 var33;
      __m256 var34;
      __m256 var35;
      __m256 var36;
      __m256 var37;
      __m256 var38;
      __m256 var39;
      __m256 var40;
      __m256 var41;
      __m256 var42;
      var26 = _mm256_broadcast_ss(&kernel[0]);
      var27 = _mm256_loadu_ps(&gray[((3 + yi) % 8) * (6 + W) + 8 * xo]);
      var25 = _mm256_mul_ps(var26, var27);
      var29 = _mm256_broadcast_ss(&kernel[1]);
      var31 = _mm256_loadu_ps(&gray[((2 + yi) % 8) * (6 + W) + 8 * xo]);
      var32 = _mm256_loadu_ps(&gray[((4 + yi) % 8) * (6 + W) + 8 * xo]);
      var30 = _mm256_add_ps(var31, var32);
      var28 = _mm256_mul_ps(var29, var30);
      var24 = _mm256_add_ps(var25, var28);
      var34 = _mm256_broadcast_ss(&kernel[2]);
      var36 = _mm256_loadu_ps(&gray[((1 + yi) % 8) * (6 + W) + 8 * xo]);
      var37 = _mm256_loadu_ps(&gray[((5 + yi) % 8) * (6 + W) + 8 * xo]);
      var35 = _mm256_add_ps(var36, var37);
      var33 = _mm256_mul_ps(var34, var35);
      var23 = _mm256_add_ps(var24, var33);
      var39 = _mm256_broadcast_ss(&kernel[3]);
      var41 = _mm256_loadu_ps(&gray[(yi % 8) * (6 + W) + 8 * xo]);
      var42 = _mm256_loadu_ps(&gray[((6 + yi) % 8) * (6 + W) + 8 * xo]);
      var40 = _mm256_add_ps(var41, var42);
      var38 = _mm256_mul_ps(var39, var40);
      var22 = _mm256_add_ps(var23, var38);
      _mm256_storeu_ps(&blur_y[8 * xo], var22);
    }
    for (int_fast32_t xo = ((13 + W) / (8)) - 1; xo < ((13 + W) / (8)); xo++) {
      __m256 var22;
      __m256 var23;
      __m256 var24;
      __m256 var25;
      __m256 var26;
      __m256 var27;
      __m256 var28;
      __m256 var29;
      __m256 var30;
      __m256 var31;
      __m256 var32;
      __m256 var33;
      __m256 var34;
      __m256 var35;
      __m256 var36;
      __m256 var37;
      __m256 var38;
      __m256 var39;
      __m256 var40;
      __m256 var41;
      __m256 var42;
      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[0]);
var26 = _mm256_blendv_ps (var26, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var27 = _mm256_maskload_ps(&gray[((3 + yi) % 8) * (6 + W) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var26, var27);
var25 = _mm256_blendv_ps (var25, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[1]);
var29 = _mm256_blendv_ps (var29, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var31 = _mm256_maskload_ps(&gray[((2 + yi) % 8) * (6 + W) + 8 * xo], cmp);
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var32 = _mm256_maskload_ps(&gray[((4 + yi) % 8) * (6 + W) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var31, var32);
var30 = _mm256_blendv_ps (var30, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var29, var30);
var28 = _mm256_blendv_ps (var28, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var25, var28);
var24 = _mm256_blendv_ps (var24, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[2]);
var34 = _mm256_blendv_ps (var34, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var36 = _mm256_maskload_ps(&gray[((1 + yi) % 8) * (6 + W) + 8 * xo], cmp);
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var37 = _mm256_maskload_ps(&gray[((5 + yi) % 8) * (6 + W) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var36, var37);
var35 = _mm256_blendv_ps (var35, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var34, var35);
var33 = _mm256_blendv_ps (var33, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var24, var33);
var23 = _mm256_blendv_ps (var23, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[3]);
var39 = _mm256_blendv_ps (var39, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var41 = _mm256_maskload_ps(&gray[(yi % 8) * (6 + W) + 8 * xo], cmp);
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var42 = _mm256_maskload_ps(&gray[((6 + yi) % 8) * (6 + W) + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var41, var42);
var40 = _mm256_blendv_ps (var40, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var39, var40);
var38 = _mm256_blendv_ps (var38, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var23, var38);
var22 = _mm256_blendv_ps (var22, add, _mm256_castsi256_ps(cmp));
}

      
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((6 - 8 * xo + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&blur_y[8 * xo], cmp, var22);
    }
    
    }
    for (int_fast32_t xo = 0; xo < ((7 + W) / (8)) - 1; xo++) {
      __m256 var43;
      __m256 var44;
      __m256 var45;
      __m256 var46;
      __m256 var47;
      __m256 var48;
      __m256 var49;
      __m256 var50;
      __m256 var51;
      __m256 var52;
      __m256 var53;
      __m256 var54;
      __m256 var55;
      __m256 var56;
      __m256 var57;
      __m256 var58;
      __m256 var59;
      __m256 var60;
      __m256 var61;
      __m256 var62;
      __m256 var63;
      __m256 var64;
      __m256 var65;
      __m256 var66;
      __m256 var67;
      __m256 var68;
      __m256 var69;
      var46 = _mm256_broadcast_ss(&two[0]);
      var47 = _mm256_loadu_ps(&gray[((3 + yi) % 8) * (6 + W) + 3 + 8 * xo]);
      var45 = _mm256_mul_ps(var46, var47);
      var52 = _mm256_broadcast_ss(&kernel[0]);
      var53 = _mm256_loadu_ps(&blur_y[3 + 8 * xo]);
      var51 = _mm256_mul_ps(var52, var53);
      var55 = _mm256_broadcast_ss(&kernel[1]);
      var57 = _mm256_loadu_ps(&blur_y[2 + 8 * xo]);
      var58 = _mm256_loadu_ps(&blur_y[4 + 8 * xo]);
      var56 = _mm256_add_ps(var57, var58);
      var54 = _mm256_mul_ps(var55, var56);
      var50 = _mm256_add_ps(var51, var54);
      var60 = _mm256_broadcast_ss(&kernel[2]);
      var62 = _mm256_loadu_ps(&blur_y[1 + 8 * xo]);
      var63 = _mm256_loadu_ps(&blur_y[5 + 8 * xo]);
      var61 = _mm256_add_ps(var62, var63);
      var59 = _mm256_mul_ps(var60, var61);
      var49 = _mm256_add_ps(var50, var59);
      var65 = _mm256_broadcast_ss(&kernel[3]);
      var67 = _mm256_loadu_ps(&blur_y[8 * xo]);
      var68 = _mm256_loadu_ps(&blur_y[6 + 8 * xo]);
      var66 = _mm256_add_ps(var67, var68);
      var64 = _mm256_mul_ps(var65, var66);
      var48 = _mm256_add_ps(var49, var64);
      var44 = _mm256_sub_ps(var45, var48);
      var69 = _mm256_loadu_ps(&gray[((3 + yi) % 8) * (6 + W) + 3 + 8 * xo]);
      var43 = _mm256_div_ps(var44, var69);
      _mm256_storeu_ps(&ratio[8 * xo], var43);
    }
    for (int_fast32_t xo = ((7 + W) / (8)) - 1; xo < ((7 + W) / (8)); xo++) {
      __m256 var43;
      __m256 var44;
      __m256 var45;
      __m256 var46;
      __m256 var47;
      __m256 var48;
      __m256 var49;
      __m256 var50;
      __m256 var51;
      __m256 var52;
      __m256 var53;
      __m256 var54;
      __m256 var55;
      __m256 var56;
      __m256 var57;
      __m256 var58;
      __m256 var59;
      __m256 var60;
      __m256 var61;
      __m256 var62;
      __m256 var63;
      __m256 var64;
      __m256 var65;
      __m256 var66;
      __m256 var67;
      __m256 var68;
      __m256 var69;
      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&two[0]);
var46 = _mm256_blendv_ps (var46, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var47 = _mm256_maskload_ps(&gray[((3 + yi) % 8) * (6 + W) + 3 + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var46, var47);
var45 = _mm256_blendv_ps (var45, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[0]);
var52 = _mm256_blendv_ps (var52, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var53 = _mm256_maskload_ps(&blur_y[3 + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var52, var53);
var51 = _mm256_blendv_ps (var51, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[1]);
var55 = _mm256_blendv_ps (var55, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var57 = _mm256_maskload_ps(&blur_y[2 + 8 * xo], cmp);
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var58 = _mm256_maskload_ps(&blur_y[4 + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var57, var58);
var56 = _mm256_blendv_ps (var56, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var55, var56);
var54 = _mm256_blendv_ps (var54, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var51, var54);
var50 = _mm256_blendv_ps (var50, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[2]);
var60 = _mm256_blendv_ps (var60, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var62 = _mm256_maskload_ps(&blur_y[1 + 8 * xo], cmp);
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var63 = _mm256_maskload_ps(&blur_y[5 + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var62, var63);
var61 = _mm256_blendv_ps (var61, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var60, var61);
var59 = _mm256_blendv_ps (var59, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var50, var59);
var49 = _mm256_blendv_ps (var49, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&kernel[3]);
var65 = _mm256_blendv_ps (var65, bc, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var67 = _mm256_maskload_ps(&blur_y[8 * xo], cmp);
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var68 = _mm256_maskload_ps(&blur_y[6 + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var67, var68);
var66 = _mm256_blendv_ps (var66, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var65, var66);
var64 = _mm256_blendv_ps (var64, mul, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps(var49, var64);
var48 = _mm256_blendv_ps (var48, add, _mm256_castsi256_ps(cmp));
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 sub = _mm256_sub_ps(var45, var48);
var44 = _mm256_blendv_ps (var44, sub, _mm256_castsi256_ps(cmp));
}

      
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var69 = _mm256_maskload_ps(&gray[((3 + yi) % 8) * (6 + W) + 3 + 8 * xo], cmp);
}

      
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 div = _mm256_div_ps(var44, var69);
var43 = _mm256_blendv_ps (var43, div, _mm256_castsi256_ps(cmp));
}

      
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-(8 * xo) + W));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&ratio[8 * xo], cmp, var43);
    }
    
    }
    for (int_fast32_t c = 0; c < 3; c++) {
      for (int_fast32_t xo = 0; xo < ((W + 7) / (8)) - 1; xo++) {
        __m256 var70;
        __m256 var71;
        __m256 var72;
        var71 = _mm256_loadu_ps(&ratio[8 * xo]);
        var72 = _mm256_loadu_ps(&input[(c) * ((H + 6) * (W + 6)) + (32 * y + yi + 3) * (W + 6) + 8 * xo + 3]);
        var70 = _mm256_mul_ps(var71, var72);
        _mm256_storeu_ps(&output[(c) * (H * W) + (32 * y + yi) * W + 8 * xo], var70);
      }
      for (int_fast32_t xo = ((W + 7) / (8)) - 1; xo < ((7 + W) / (8)); xo++) {
        __m256 var70;
        __m256 var71;
        __m256 var72;
        
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((-8 * xo + W + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var71 = _mm256_maskload_ps(&ratio[8 * xo], cmp);
}

        
{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((W + -8 * xo + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    var72 = _mm256_maskload_ps(&input[(c) * ((H + 6) * (W + 6)) + (32 * y + yi + 3) * (W + 6) + 8 * xo + 3], cmp);
}

        
{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32((W + -8 * xo + 0));
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps(var71, var72);
var70 = _mm256_blendv_ps (var70, mul, _mm256_castsi256_ps(cmp));
}

        
    {
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32((W + -8 * xo + 0));
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&output[(c) * (H * W) + (32 * y + yi) * W + 8 * xo], cmp, var70);
    }
    
      }
    }
  }
  free(blur_y);
  free(ratio);
  free(gray);
}
free(two);
free(kernel);
free(rgb);
}


/* relying on the following instruction..."
mm256_add_ps(out,x,y)
{out_data} = _mm256_add_ps({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
*/

/* relying on the following instruction..."
mm256_div_ps(out,x,y)
{out_data} = _mm256_div_ps({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_loadu_ps(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
*/

/* relying on the following instruction..."
mm256_mul_ps(out,x,y)
{out_data} = _mm256_mul_ps({x_data}, {y_data});
*/

/* relying on the following instruction..."
mm256_prefix_add_ps(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 add = _mm256_add_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, add, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_broadcast_ss(out,val,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 bc = _mm256_broadcast_ss(&{val_data});
{out_data} = _mm256_blendv_ps ({out_data}, bc, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_div_ps(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 div = _mm256_div_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, div, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_load_ps(dst,src,bound)

{{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    {dst_data} = _mm256_maskload_ps(&{src_data}, cmp);
}}

*/

/* relying on the following instruction..."
mm256_prefix_mul_ps(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 mul = _mm256_mul_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, mul, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
mm256_prefix_store_ps(dst,src,bound)

    {{
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i prefix = _mm256_set1_epi32({bound});
    __m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
    _mm256_maskstore_ps(&{dst_data}, cmp, {src_data});
    }}
    
*/

/* relying on the following instruction..."
mm256_prefix_sub_ps(out,x,y,bound)

{{
__m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
__m256i prefix = _mm256_set1_epi32({bound});
__m256i cmp = _mm256_cmpgt_epi32(prefix, indices);
__m256 sub = _mm256_sub_ps({x_data}, {y_data});
{out_data} = _mm256_blendv_ps ({out_data}, sub, _mm256_castsi256_ps(cmp));
}}

*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
mm256_sub_ps(out,x,y)
{out_data} = _mm256_sub_ps({x_data}, {y_data});
*/
