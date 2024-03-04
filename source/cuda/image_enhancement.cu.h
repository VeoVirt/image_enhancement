#include <stdint.h>

#ifndef IMAGE_ENHANCEMENT_H
#define IMAGE_ENHANCEMENT_H

#define KERNEL_RADIUS 14
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


extern "C"
__global__ void convolutionRowsGPU(float *d_Dst, float *d_Src, int imageW,
                                   int imageH, int pitch);

extern "C"
__global__ void convolutionColumnsGPU(float *d_Dst, float *d_Src, int imageW,
                                      int imageH, int pitch);

extern "C"
__global__ void scale(uint8_t* src, float* dst, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_ud(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_du(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_lr(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_rl(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void enhance_image(
    uint8_t* image, float* ph_mask, float threshold_dark_tones, float local_boost, float saturation_degree,
    float mid_tone_mapped, float tonal_width_mapped, float areas_dark_mapped, float areas_bright_mapped, float detail_amp_global, uint32_t width, uint32_t height
);

#endif