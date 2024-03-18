#include <cstdint>
#include <cuda_runtime.h>
#include "tone_mapping.cuh"
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

#define COLUMNS_BLOCKDIM_X 8
#define COLUMNS_BLOCKDIM_Y 4
#define COLUMNS_RESULT_STEPS 4
#define COLUMNS_HALO_STEPS 4

#define KERNEL_RADIUS 14
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


//extern "C"
//__global__ void convolutionKernel(float* d_Dst, cudaSurfaceObject_t d_Src, int imageW,
//                                      int imageH, int pitch) {
//  // Handle to thread block group
//  cg::thread_block cta = cg::this_thread_block();
//  __shared__ uint8_t
//      s_Data[ROWS_BLOCKDIM_Y * 2*ROWS_HALO_STEPS][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) *
//                              ROWS_BLOCKDIM_X];
//
//  //__shared__ float c_Kernel[KERNEL_LENGTH];
//
//  // Offset to the left halo edge
//  const int baseX =
//      (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X +
//      threadIdx.x;
//  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
//
//  d_Dst += baseY * pitch + baseX;
//
//
//for (int j = 0; i < ROWS_HALO_STEPS; i++)
//// Load main data
//#pragma unroll
//
//  for (int i = 0; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
//    uint8_t elem;
//    surf2Dread(&elem, d_Src, (baseX+i*ROWS_BLOCKDIM_X), baseY);
//    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = elem;
//        //d_Src[i * ROWS_BLOCKDIM_X];
//  }
//
//// Load left halo
//#pragma unroll
//
//  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
//    uint8_t elem_left;
//    surf2Dread(&elem_left, d_Src, (baseX+i*ROWS_BLOCKDIM_X), baseY, cudaBoundaryModeClamp);
//    //(baseX >= -i * ROWS_BLOCKDIM_X) ?
//    //    surf2Dread(&elem_left, d_Src, (baseX+i*ROWS_BLOCKDIM_X), baseY) :
//    //        surf2Dread(&elem_left, d_Src, 0, baseY);
//    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = elem_left;
//  }
//
//// Load right halo
//#pragma unroll
//
//  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
//       i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
//    uint8_t elem_right;
//    surf2Dread(&elem_right, d_Src, (baseX+i*ROWS_BLOCKDIM_X), baseY, cudaBoundaryModeClamp);
//    //(imageW - baseX > i * ROWS_BLOCKDIM_X) ?
//    //    surf2Dread(&elem_right, d_Src, (baseX+i*ROWS_BLOCKDIM_X), baseY, cudaBoundaryModeClamp) :
//    //        surf2Dread(&elem_right, d_Src, (pitch-1), baseY);
//    s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = elem_right;
//  }
//
//  // Compute and store results
//  cg::sync(cta);
//  static const float c_Kernel[29] = { 0.00801895, 0.01056259, 0.013632, 0.01723796, 0.02135743, 0.0259268,
//                            0.03083797, 0.03593846, 0.04103646, 0.04591105, 0.05032705, 0.05405333,
//                            0.05688272, 0.05865096, 0.0592525 , 0.05865096, 0.05688272, 0.05405333,
//                            0.05032705, 0.04591105, 0.04103646, 0.03593846, 0.03083797, 0.0259268,
//                            0.02135743, 0.01723796, 0.013632, 0.01056259, 0.00801895 };
//#pragma unroll
//
//  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
//    float sum = 0;
//
//#pragma unroll
//
//    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
//      sum += c_Kernel[KERNEL_RADIUS - j] *
//             ((float)(s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j]) / 255.0f);
//    }
//
//    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
//    //surf2Dwrite(sum,d_Dst,(i * ROWS_BLOCKDIM_X + baseX)*sizeof(float),baseY);
//  }
//}

extern "C"
__global__ void convolutionRowsKernel(float* d_Dst, uint8_t d_Src, int imageW,
                                      int imageH, int pitch) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint8_t
      s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) *
                              ROWS_BLOCKDIM_X];

  //__shared__ float c_Kernel[KERNEL_LENGTH];

  // Offset to the left halo edge
  const int baseX =
      (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X +
      threadIdx.x;
  const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

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
  //static const float c_Kernel[17] = { 0.03083797, 0.03593846, 0.04103646, 0.04591105, 0.05032705, 0.05405333,
  //                          0.05688272, 0.05865096, 0.0592525 , 0.05865096, 0.05688272, 0.05405333,
  //                          0.05032705, 0.04591105, 0.04103646, 0.03593846, 0.03083797 };
#pragma unroll

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    float sum = 0;

#pragma unroll

    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      sum += c_Kernel[KERNEL_RADIUS - j] *
             ((float)(s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j]) / 255.0f);
    }

    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////

extern "C"
__global__ void convolutionColumnsKernel(float *d_Dst, float* d_Src, int imageW,
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
  //static const float c_Kernel[17] = { 0.03083797, 0.03593846, 0.04103646, 0.04591105, 0.05032705, 0.05405333,
  //                          0.05688272, 0.05865096, 0.0592525 , 0.05865096, 0.05688272, 0.05405333,
  //                          0.05032705, 0.04591105, 0.04103646, 0.03593846, 0.03083797 };
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

__device__ float change_color_saturation_uv(
    float value, float mask, float threshold_dark_tones, float local_boost, float saturation_degree
){
    float detail_amplification_local = ((1 - min(1.0f, mask / threshold_dark_tones)) * local_boost) + 1;
    return max(-1.0f, min(1.0f, value * saturation_degree * detail_amplification_local));
}

// play with bindings/datatypes/reuse/operators..
extern "C"
__global__ void enhance_image(
    uint8_t* Y, float* ph_mask, float threshold_dark_tones, float local_boost, float saturation_degree,
    float mid_tone_mapped, float tonal_width_mapped, float areas_dark_mapped, float areas_bright_mapped, float detail_amp_global, uint32_t width, uint32_t height
){
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (width <= x || height <= y){
        return;
    }

    float mask = ph_mask[y * width + x];

    uint8_t gray_raw = Y[y * width + x];
    
    //surf2Dread(&gray_raw,Y,x,y);
    float gray = (float) gray_raw / 255.0f;
    gray = local_contrast_enhancement(gray, mask, threshold_dark_tones, local_boost, detail_amp_global);
    gray = spatial_tonemapping(
        gray, mask, mid_tone_mapped, tonal_width_mapped, areas_dark_mapped,
        areas_bright_mapped
    );
    gray_raw = (uint8_t) (max(0.0f, min(255.0f, gray*255.0)));
    Y[y * width + x] = gray_raw;
    //surf2Dwrite(gray_raw, Y, x, y);
    

    //float u = 2*(((float) U[y * width + x]) / 255.0f) - 1;
    //float v = 2*(((float) V[y * width + x]) / 255.0f) - 1;
    // hmmm color is a bit dull, and running time is 0.35028000056743624 compared to 0.20
    //U[y*width + x] = (uint8_t) max(0.0f, min(255.0f, (change_color_saturation_uv(u, mask, threshold_dark_tones, local_boost, saturation_degree)+1)/2 * 255.0f));
    //V[y*width + x] = (uint8_t) max(0.0f, min(255.0f, (change_color_saturation_uv(v, mask, threshold_dark_tones, local_boost, saturation_degree)+1)/2 * 255.0f));
}
