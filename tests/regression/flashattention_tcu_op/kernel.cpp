#include "common.h"

#include <VX_config.h>
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::fp16, vt::fp32>;

static constexpr uint32_t kDescQ = 0;
static constexpr uint32_t kDescK = 1;
#if FA_SPARSITY != 2
static constexpr uint32_t kDescP = 2;
#endif
static constexpr uint32_t kDescV = 3;
#if FA_SPARSITY == 2
static constexpr uint32_t kDescQBitmap = 4;
static constexpr uint32_t kDescKBitmap = 5;
static constexpr uint32_t kDescVBitmap = 6;
#endif
static constexpr uint32_t kInputTileElements = kTile * kHeadDimension;
static constexpr uint32_t kStageBytes = kLocalMemoryBytes / 2;
#ifndef FA_SPARSITY
#define FA_SPARSITY 0
#endif

#if FA_SPARSITY == 2
static constexpr uint32_t kBitmapWords = kInputTileElements / 32;
#endif

#undef __local_mem
#define __local_mem(size) (void *)(csr_read(VX_CSR_CTA_LMEM_ADDR))

static inline __attribute__((always_inline)) void run_mma_op(
    uint32_t *a_lmem, uint32_t *b_lmem, float *destination,
    uint32_t completion_barrier, uint32_t *a_bitmap = nullptr,
    uint32_t *b_bitmap = nullptr, uint32_t max_a_blocks = 0,
    uint32_t max_b_blocks = 0, uint32_t sparsity = 0) {
  const uintptr_t rs1 = (uintptr_t)vx_wgather(
      (size_t)(uintptr_t)a_lmem, (size_t)(uintptr_t)b_lmem,
      (size_t)(uintptr_t)nullptr, (size_t)(uintptr_t)destination);
  const uintptr_t rs2 = (uintptr_t)vx_wgather(
      (size_t)(uintptr_t)a_bitmap, (size_t)(uintptr_t)b_bitmap,
      (size_t)completion_barrier,
      (size_t)((kHeadDimension << 24) | (max_a_blocks << 18) |
               (max_b_blocks << 12) | (vt::fp32::id << 8) |
               (vt::fp16::id << 4) | (sparsity << 2) | 3u));
  ctx::mma_op(rs1, rs2);
}

extern "C" void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  static_assert(LMEM_ENABLED);
  const uint32_t tile_id = blockIdx.x;
  const uint32_t row = threadIdx.x;
  const uint32_t barrier_base = csr_read(VX_CSR_CTA_ID);
  const bool __UNIFORM__ is_dxa_warp = csr_read(VX_CSR_CTA_RANK) == 0;

  auto scores = reinterpret_cast<float *>(arg->scores_addr) +
                tile_id * kAttentionGroups * kTile * kAttentionGroupSize;
  auto probabilities = reinterpret_cast<uint16_t *>(arg->probabilities_addr) +
                       tile_id * kAttentionGroups * kTile *
                           kAttentionGroupSize;
  auto pv = reinterpret_cast<float *>(arg->pv_addr) +
            tile_id * kAttentionGroups * kTile * kHeadDimension;
  auto output = reinterpret_cast<float *>(arg->output_addr) +
                tile_id * kTile * kHeadDimension;

  auto lmem = reinterpret_cast<uint32_t *>(__local_mem(kLocalMemoryBytes));
  uint32_t *stage0_a = lmem;
  uint32_t *stage0_b = stage0_a + kInputTileElements / 2;
  uint32_t *stage1_a = lmem + kStageBytes / sizeof(uint32_t);
  uint32_t *stage1_b = stage1_a + kInputTileElements / 2;

#if FA_SPARSITY == 2
  uint32_t *stage0_q_bitmap = stage0_a;
  uint32_t *stage1_q_bitmap = stage1_a;
  stage0_a += kBitmapWords;
  stage1_a += kBitmapWords;
  uint32_t *stage0_k_bitmap = lmem + kInputTileElements / 2 + 16;
  uint32_t *stage1_k_bitmap =
      lmem + kStageBytes / sizeof(uint32_t) + kInputTileElements / 2 + 16;
  stage0_b = stage0_k_bitmap + kBitmapWords + 16;
  stage1_b = stage1_k_bitmap + kBitmapWords + 16;
#endif

#if FA_SPARSITY != 2
  float running_max = -3.402823466e+38f;
  float running_sum = 0.0f;
#else
  float group_maxima[kAttentionGroups];
  float group_sums[kAttentionGroups];
#endif
#if FA_SPARSITY != 2
  if (row < kTile) {
    for (uint32_t column = 0; column < kHeadDimension; ++column)
      output[row * kHeadDimension + column] = 0.0f;
  }
  __syncthreads();
#endif

  for (uint32_t group = 0; group < kAttentionGroups; ++group) {
#if FA_SPARSITY == 2
    stage0_a = stage0_q_bitmap + kBitmapWords;
    stage1_a = stage1_q_bitmap + kBitmapWords;
    stage0_b = stage0_k_bitmap + kBitmapWords + 16;
    stage1_b = stage1_k_bitmap + kBitmapWords + 16;
#endif
    float *group_scores = scores + group * kTile * kAttentionGroupSize;
    uint16_t *group_probabilities =
        probabilities + group * kTile * kAttentionGroupSize;
#if FA_SPARSITY == 2
    group_probabilities = reinterpret_cast<uint16_t *>(lmem);
#endif
    float *group_pv = pv + group * kTile * kHeadDimension;

    const uint32_t load0 = barrier_base;
    const uint32_t load1 = barrier_base + (1u << 8);
    const uint32_t done0 = barrier_base + (2u << 8);
    const uint32_t done1 = barrier_base + (3u << 8);

    vx_barrier(done0, 1);
    if (is_dxa_warp) {
#if FA_SPARSITY == 2
      vx_dxa_issue_2d_wg(kDescQBitmap, load0, stage0_q_bitmap, 0, tile_id);
      vx_dxa_issue_2d_wg(kDescKBitmap, load0, stage0_k_bitmap, 0, group * 2);
#else
#endif
      vx_dxa_issue_2d_wg(kDescQ, load0, stage0_a, 0, tile_id);
      vx_dxa_issue_2d_wg(kDescK, load0, stage0_b, 0, group * 2);
    }
    vx_barrier(load0, 1);
    if (is_dxa_warp)
#if FA_SPARSITY == 2
      run_mma_op(stage0_a, stage0_b, group_scores, done0,
                 stage0_q_bitmap, stage0_k_bitmap, arg->max_q_blocks,
                 arg->max_k_blocks, 2);
#else
      run_mma_op(stage0_a, stage0_b, group_scores, done0);
#endif

    vx_barrier(done1, 1);
    if (is_dxa_warp) {
#if FA_SPARSITY == 2
      vx_dxa_issue_2d_wg(kDescQBitmap, load1, stage1_q_bitmap, 0, tile_id);
      vx_dxa_issue_2d_wg(kDescKBitmap, load1, stage1_k_bitmap, 0,
                         group * 2 + 1);
#else
#endif
      vx_dxa_issue_2d_wg(kDescQ, load1, stage1_a, 0, tile_id);
      vx_dxa_issue_2d_wg(kDescK, load1, stage1_b, 0, group * 2 + 1);
    }

    vx_barrier(done0, 1);
    vx_barrier(load1, 1);
    if (is_dxa_warp)
#if FA_SPARSITY == 2
      run_mma_op(stage1_a, stage1_b, group_scores + kTileElements, done1,
                 stage1_q_bitmap, stage1_k_bitmap, arg->max_q_blocks,
                 arg->max_k_blocks, 2);
#else
      run_mma_op(stage1_a, stage1_b, group_scores + kTileElements, done1);
#endif

    float group_max = -3.402823466e+38f;
    if (row < kTile) {
      for (uint32_t key = 0; key < kTile; ++key) {
        const uint32_t index = row * kTile + key;
        const float score = group_scores[index] * 0.125f;
        group_max = score > group_max ? score : group_max;
      }
    }
    vx_barrier(done1, 1);

    if (row < kTile) {
      for (uint32_t key = kTile; key < kAttentionGroupSize; ++key) {
        const uint32_t index = kTileElements + row * kTile + key - kTile;
        const float score = group_scores[index] * 0.125f;
        group_max = score > group_max ? score : group_max;
      }
    }

#if FA_SPARSITY != 2
    float previous_scale = 0.0f;
#endif
    if (row < kTile) {
#if FA_SPARSITY == 2
      const float new_max = group_max;
#else
      const float new_max = running_max > group_max ? running_max : group_max;
      previous_scale = running_sum == 0.0f
                           ? 0.0f
                           : exp2_taylor(running_max - new_max);
#endif
      float group_sum = 0.0f;
      for (uint32_t key = 0; key < kAttentionGroupSize; ++key) {
        const uint32_t score_index =
            (key / kTile) * kTileElements + row * kTile + key % kTile;
        const float probability =
            exp2_taylor(group_scores[score_index] * 0.125f - new_max);
        group_probabilities[key * kTile + row] =
            float_to_fp16(probability);
        group_sum += probability;
      }
#if FA_SPARSITY == 2
      group_maxima[group] = new_max;
      group_sums[group] = group_sum;
#else
      running_sum = running_sum * previous_scale + group_sum;
      running_max = new_max;
#endif
    }
    __syncthreads();

    vx_barrier(done0, 1);
#if FA_SPARSITY == 2
    stage0_a = lmem;
    stage1_a = lmem;
    uint32_t *stage0_v_bitmap = stage0_a + kInputTileElements / 2;
    uint32_t *stage1_v_bitmap =
        lmem + kStageBytes / sizeof(uint32_t) + kInputTileElements / 2;
    stage0_b = stage0_v_bitmap + kBitmapWords;
    stage1_b = stage1_v_bitmap + kBitmapWords;
#endif
    if (is_dxa_warp) {
#if FA_SPARSITY == 2
      vx_dxa_issue_2d_wg(kDescVBitmap, load0, stage0_v_bitmap, 0,
                         group * kOutputTiles);
#else
      vx_dxa_issue_2d_wg(kDescP, load0, stage0_a, 0,
                         tile_id * kAttentionGroups + group);
#endif
      vx_dxa_issue_2d_wg(kDescV, load0, stage0_b, 0,
                         group * kOutputTiles);
    }
    vx_barrier(load0, 1);
    if (is_dxa_warp)
#if FA_SPARSITY == 2
      run_mma_op(stage0_a, stage0_b, group_pv, done0, nullptr,
                 stage0_v_bitmap, 0, arg->max_v_blocks, 1);
#else
      run_mma_op(stage0_a, stage0_b, group_pv, done0);
#endif

    vx_barrier(done1, 1);
    if (is_dxa_warp) {
#if FA_SPARSITY == 2
      vx_dxa_issue_2d_wg(kDescVBitmap, load1, stage1_v_bitmap, 0,
                         group * kOutputTiles + 1);
#else
      vx_dxa_issue_2d_wg(kDescP, load1, stage1_a, 0,
                         tile_id * kAttentionGroups + group);
#endif
      vx_dxa_issue_2d_wg(kDescV, load1, stage1_b, 0,
                         group * kOutputTiles + 1);
    }

    vx_barrier(done0, 1);
    vx_barrier(load1, 1);
    if (is_dxa_warp)
#if FA_SPARSITY == 2
      run_mma_op(stage1_a, stage1_b, group_pv + kTileElements, done1,
                 nullptr, stage1_v_bitmap, 0, arg->max_v_blocks, 1);
#else
      run_mma_op(stage1_a, stage1_b, group_pv + kTileElements, done1);
#endif

#if FA_SPARSITY != 2
    if (row < kTile) {
      for (uint32_t column = 0; column < kTile; ++column) {
        const uint32_t index = row * kHeadDimension + column;
        output[index] = output[index] * previous_scale +
                        group_pv[row * kTile + column];
      }
    }
#endif
    vx_barrier(done1, 1);

#if FA_SPARSITY == 2
    if (row < kTile) {
      if (group == 0 && kAttentionGroups != 1) {
        for (uint32_t column = 0; column < kHeadDimension; ++column) {
          const uint32_t pv_index =
              (column / kTile) * kTileElements + row * kTile + column % kTile;
          output[row * kHeadDimension + column] = group_pv[pv_index];
        }
      } else if (group == kAttentionGroups - 1) {
        float final_max = group_maxima[0];
        for (uint32_t index = 1; index < kAttentionGroups; ++index)
          final_max = group_maxima[index] > final_max
                          ? group_maxima[index]
                          : final_max;
        float denominator = 0.0f;
        float group_scales[kAttentionGroups];
        for (uint32_t index = 0; index < kAttentionGroups; ++index) {
          group_scales[index] =
              exp2_taylor(group_maxima[index] - final_max);
          denominator += group_sums[index] * group_scales[index];
        }
        const float inverse_sum = 1.0f / denominator;
        for (uint32_t column = 0; column < kHeadDimension; ++column) {
          const uint32_t pv_index =
              (column / kTile) * kTileElements + row * kTile + column % kTile;
          float numerator = group_pv[pv_index] * group_scales[group];
          if (group != 0)
            numerator += output[row * kHeadDimension + column] *
                         group_scales[0];
          output[row * kHeadDimension + column] = numerator * inverse_sum;
        }
      }
    }
#else
    if (row < kTile) {
      for (uint32_t column = kTile; column < kHeadDimension; ++column) {
        const uint32_t output_index = row * kHeadDimension + column;
        const uint32_t pv_index =
            kTileElements + row * kTile + column - kTile;
        output[output_index] = output[output_index] * previous_scale +
                               group_pv[pv_index];
      }
    }
    __syncthreads();
#endif
  }

  if (row < kTile) {
#if FA_SPARSITY != 2
    const float inverse_sum = 1.0f / running_sum;
    for (uint32_t column = 0; column < kHeadDimension; ++column)
      output[row * kHeadDimension + column] *= inverse_sum;
#endif
  }
}
