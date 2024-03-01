#include <cstdint>
#include <cuda_runtime.h>
#include "image_enhancement.cu.h"
#include <assert.h>
//#include <helper_cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define LUT_RES 256
#define EPSILON 1.0f / 256.0f

#define ROWS_BLOCKDIM_X 8
#define ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 4
#define ROWS_HALO_STEPS 2

#define COLUMNS_BLOCKDIM_X 4
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 2
#define COLUMNS_HALO_STEPS 2

//__constant__ float c_Kernel[KERNEL_LENGTH]

__device__ float to_gray(float rgb[3]){
    return rgb[0] * 0.2125f + rgb[1] * 0.7154f + rgb[2] * 0.0721f;
}

extern "C"
__global__ void color_to_gray(uint8_t* color, float* gray, uint32_t width, uint32_t height){
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (width <= x || height <= y){
        return;
    }

    float rgb[3];
    rgb[0] = ((float) color[y * width * 3 + x * 3 + 0]) / 255.0f;
    rgb[1] = ((float) color[y * width * 3 + x * 3 + 1]) / 255.0f;
    rgb[2] = ((float) color[y * width * 3 + x * 3 + 2]) / 255.0f;

    gray[y * width + x] = to_gray(rgb);
}

extern "C"
__global__ void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW,
                                      int imageH, int pitch) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float
      s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) *
                              ROWS_BLOCKDIM_X];

  //__shared__ float c_Kernel[KERNEL_LENGTH];

  // Offset to the left halo edge
  const int baseX =
      (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X +
      threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

  const float * d_Org = d_Src;
  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        d_Src[i * ROWS_BLOCKDIM_X];
  }

// Load left halo
#pragma unroll

  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : d_Org[baseY*pitch];
  }

// Load right halo
#pragma unroll

  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
       i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
        (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : d_Org[(baseY+1)*(pitch)-1];
  }

  // Compute and store results
  cg::sync(cta);
  static const float c_Kernel[29] = { 0.00801895, 0.01056259, 0.013632, 0.01723796, 0.02135743, 0.0259268,
                            0.03083797, 0.03593846, 0.04103646, 0.04591105, 0.05032705, 0.05405333,
                            0.05688272, 0.05865096, 0.0592525 , 0.05865096, 0.05688272, 0.05405333,
                            0.05032705, 0.04591105, 0.04103646, 0.03593846, 0.03083797, 0.0259268,
                            0.02135743, 0.01723796, 0.013632, 0.01056259, 0.00801895 };
#pragma unroll

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    float sum = 0;

#pragma unroll

    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      sum += c_Kernel[KERNEL_RADIUS - j] *
             s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
    }

    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

//extern "C" void convolutionRowsGPU(float *d_Dst, float *d_Src, int imageW,
//                                   int imageH) {
//  assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
//  assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
//  assert(imageH % ROWS_BLOCKDIM_Y == 0);
//
//  dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X),
//              imageH / ROWS_BLOCKDIM_Y);
//  dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
//
//  convolutionRowsKernel<<<blocks, threads>>>(d_Dst, d_Src, imageW, imageH,
//                                             imageW);
//  getLastCudaError("convolutionRowsKernel() execution failed\n");
//}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////

extern "C"
__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW,
                                         int imageH, int pitch) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS +
                                               2 * COLUMNS_HALO_STEPS) *
                                                  COLUMNS_BLOCKDIM_Y +
                                              1];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
  const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) *
                        COLUMNS_BLOCKDIM_Y +
                    threadIdx.y;
  const float* d_Org = d_Src;
  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
  }

// Upper halo
#pragma unroll

  for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (baseY >= -i * COLUMNS_BLOCKDIM_Y)
            ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch]
            : d_Org[baseX];
  }

// Lower halo
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS;
       i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
        (imageH - baseY > i * COLUMNS_BLOCKDIM_Y)
            ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch]
            : d_Org[imageW*(imageH-1)+baseX];
  }

  // Compute and store results
  cg::sync(cta);
  static const float c_Kernel[29] = { 0.00801895, 0.01056259, 0.013632, 0.01723796, 0.02135743, 0.0259268,
                            0.03083797, 0.03593846, 0.04103646, 0.04591105, 0.05032705, 0.05405333,
                            0.05688272, 0.05865096, 0.0592525 , 0.05865096, 0.05688272, 0.05405333,
                            0.05032705, 0.04591105, 0.04103646, 0.03593846, 0.03083797, 0.0259268,
                            0.02135743, 0.01723796, 0.013632, 0.01056259, 0.00801895 };

#pragma unroll
  for (int i = COLUMNS_HALO_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    float sum = 0;
#pragma unroll
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      sum += c_Kernel[KERNEL_RADIUS - j] *
             s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
    }

    d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
  }
}

//extern "C" void convolutionColumnsGPU(float *d_Dst, float *d_Src, int imageW,
//                                      int imageH) {
//  assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
//  assert(imageW % COLUMNS_BLOCKDIM_X == 0);
//  assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);
//
//  dim3 blocks(imageW / COLUMNS_BLOCKDIM_X,
//              imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
//  dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
//
//  convolutionColumnsKernel<<<blocks, threads>>>(d_Dst, d_Src, imageW, imageH,
//                                                imageW);
//  getLastCudaError("convolutionColumnsKernel() execution failed\n");
//}

extern "C"
__global__ void photometric_mask_ud(float* ph_mask, float* lut, uint32_t width, uint32_t height){
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (width <= j){
        return;
    }

    for (uint32_t i = 0; i < height - 2; ++i){
        float diff = abs(ph_mask[i * width + j] - ph_mask[(i + 2) * width + j]);
        float sigmoid = lut[(uint8_t)(diff * (LUT_RES - 1))];
        ph_mask[(i + 1) * width + j] = ph_mask[(i + 1) * width + j] * sigmoid + ph_mask[i * width + j] * (1 - sigmoid);
    }
}

extern "C"
__global__ void photometric_mask_du(float* ph_mask, float* lut, uint32_t width, uint32_t height){
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (width <= j){
        return;
    }

    for (uint32_t i = height - 2; i > 1; --i){
        float diff = abs(ph_mask[(i - 1) * width + j] - ph_mask[(i + 1) * width + j]);
        float sigmoid = lut[(uint8_t)(diff * (LUT_RES - 1))];
        ph_mask[i * width + j] = ph_mask[i * width + j] * sigmoid + ph_mask[(i + 1) * width + j] * (1 - sigmoid);
    }
}

extern "C"
__global__ void photometric_mask_lr(float* ph_mask, float* lut, uint32_t width, uint32_t height){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (height <= i){
        return;
    }

    for (uint32_t j = 0; j < width - 2; ++j){
        float diff = abs(ph_mask[i * width + j] - ph_mask[i * width + j + 2]);
        float sigmoid = lut[(uint8_t)(diff * (LUT_RES - 1))];
        ph_mask[i * width + j + 1] = ph_mask[i * width + j + 1] * sigmoid + ph_mask[i * width + j] * (1 - sigmoid);
    }
}

extern "C"
__global__ void photometric_mask_rl(float* ph_mask, float* lut, uint32_t width, uint32_t height){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (height <= i){
        return;
    }

    for (uint32_t j = width - 2; j > 1; --j){
        float diff = abs(ph_mask[i * width + j - 1] - ph_mask[i * width + j + 1]);
        float sigmoid = lut[(uint8_t)(diff * (LUT_RES - 1))];
        ph_mask[i * width + j] = ph_mask[i * width + j] * sigmoid + ph_mask[i * width + j + 1] * (1 - sigmoid);
    }
}

__device__ float local_contrast_enhancement(
    float gray, float mask, float threshold_dark_tones, float local_boost,
    float detail_amp_global
){
    float detail_amp_local = min(mask / threshold_dark_tones, 1.0f);
    detail_amp_local = (1 - detail_amp_local) * local_boost + 1;

    float value = mask + (gray - mask) * detail_amp_global * detail_amp_local;

    return max(0.0f, min(1.0f, value));
}

__device__ float spatial_tonemapping(
    float value, float mask, float mid_tone_mapped, float tonal_width_mapped, float areas_dark_mapped,
    float areas_bright_mapped
){
    float alpha;
    float tone_continuation_factor;

    float mask_inv = 1 - mask;

    float lower = value * (value < mid_tone_mapped);
    float upper = value * (value >= mid_tone_mapped);

    alpha = (mask * mask) / tonal_width_mapped;
    tone_continuation_factor = mid_tone_mapped / (mid_tone_mapped + EPSILON - mask);
    alpha = alpha * tone_continuation_factor + areas_dark_mapped;
    lower = (lower * (alpha + 1)) / (alpha + lower);

    alpha = (mask_inv * mask_inv) / tonal_width_mapped;
    tone_continuation_factor = mid_tone_mapped / ((1 - mid_tone_mapped) - mask_inv);
    alpha = alpha * tone_continuation_factor + areas_bright_mapped;
    upper = (upper * alpha) / (alpha + 1 - upper);

    return lower + upper;
}

__device__ float srgb_to_linear(float value){
    float lower = value * (value <= 0.04045f) / 12.92f;
    float upper = powf((value + 0.055f) * (value > 0.04045f) / 1.055f, 2.4f);

    return lower + upper;
}

__device__ float linear_to_srgb(float value){
    float lower = value * (value <= 0.0031308f) * 12.92f;
    float upper = powf(value * (value > 0.0031308f), 1.0f / 2.4f) * 1.055f - 0.055f;

    return max(0.0f, min(1.0f, lower + upper));
}

__device__ void graytone_to_color(float rgb[3], float gray){
    rgb[0] = srgb_to_linear(rgb[0]);
    rgb[1] = srgb_to_linear(rgb[1]);
    rgb[2] = srgb_to_linear(rgb[2]);

    float graytone_linear = srgb_to_linear(gray);

    float gray_linear = to_gray(rgb);
    if (gray_linear <= 0.0f){
        gray_linear = EPSILON;
    }

    float tone_ratio = graytone_linear / gray_linear;

    rgb[0] = max(0.0f, min(1.0f, rgb[0] * tone_ratio));
    rgb[1] = max(0.0f, min(1.0f, rgb[1] * tone_ratio));
    rgb[2] = max(0.0f, min(1.0f, rgb[2] * tone_ratio));

    rgb[0] = linear_to_srgb(rgb[0]);
    rgb[1] = linear_to_srgb(rgb[1]);
    rgb[2] = linear_to_srgb(rgb[2]);
}

__device__ void change_color_saturation(
    float rgb[3], float mask, float threshold_dark_tones, float local_boost, float saturation_degree
){
    float gray = (rgb[0] + rgb[1] + rgb[2]) / 3.0f;

    rgb[0] = rgb[0] - gray;
    rgb[1] = rgb[1] - gray;
    rgb[2] = rgb[2] - gray;

    float detail_amplification_local = ((1 - min(1.0f, mask / threshold_dark_tones)) * local_boost) + 1;

    rgb[0] = max(0.0f, min(1.0f, gray + rgb[0] * saturation_degree * detail_amplification_local));
    rgb[1] = max(0.0f, min(1.0f, gray + rgb[1] * saturation_degree * detail_amplification_local));
    rgb[2] = max(0.0f, min(1.0f, gray + rgb[2] * saturation_degree * detail_amplification_local));
}

// play with bindings/datatypes/reuse/operators..
extern "C"
__global__ void enhance_image(
    uint8_t* image, float* ph_mask, float threshold_dark_tones, float local_boost, float saturation_degree,
    float mid_tone_mapped, float tonal_width_mapped, float areas_dark_mapped, float areas_bright_mapped, float detail_amp_global, uint32_t width, uint32_t height
){
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (width <= x || height <= y){
        return;
    }

    float rgb[3];
    uint32_t idx = y * width * 3 + x * 3;
    rgb[0] = ((float) image[idx + 0]) / 255.0f;
    rgb[1] = ((float) image[idx + 1]) / 255.0f;
    rgb[2] = ((float) image[idx + 2]) / 255.0f;

    float mask = ph_mask[y * width + x];

    float gray;
    gray = to_gray(rgb);
    gray = local_contrast_enhancement(gray, mask, threshold_dark_tones, local_boost, detail_amp_global);
    gray = spatial_tonemapping(
        gray, mask, mid_tone_mapped, tonal_width_mapped, areas_dark_mapped,
        areas_bright_mapped
    );

    graytone_to_color(rgb, gray);

    change_color_saturation(rgb, mask, threshold_dark_tones, local_boost, saturation_degree);

    image[idx + 0] = (uint8_t) max(0.0f, min(255.0f, rgb[0] * 255.0f));
    image[idx + 1] = (uint8_t) max(0.0f, min(255.0f, rgb[1] * 255.0f));
    image[idx + 2] = (uint8_t) max(0.0f, min(255.0f, rgb[2] * 255.0f));
}
