#include <cstdint>
#include <stdint.h>

#ifndef IMAGE_ENHANCEMENT_H
#define IMAGE_ENHANCEMENT_H

extern "C"
__global__ void convolutionRowsKernel(float *d_Dst, uint8_t* d_Src, int imageW,
                                      int imageH, int pitch);


extern "C"
__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW,
                                         int imageH, int pitch);

extern "C"
__global__ void enhance_image(
    uint8_t* Y, float* ph_mask, float threshold_dark_tones, float local_boost, float saturation_degree,
    float mid_tone_mapped, float tonal_width_mapped, float areas_dark_mapped, float areas_bright_mapped, float detail_amp_global, uint32_t width, uint32_t height
);

#endif