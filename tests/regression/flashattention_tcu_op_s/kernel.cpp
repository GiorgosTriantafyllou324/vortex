#include "../flashattention_tcu_op/common.h"

#include <VX_config.h>
#include <vx_intrinsics.h>
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::fp16, vt::fp32>;

static constexpr uint32_t kDescQ = 0;
static constexpr uint32_t kDescK = 1;
static constexpr uint32_t kDescP = 2;
static constexpr uint32_t kDescV = 3;
static constexpr uint32_t kDescQBitmap = 4;
static constexpr uint32_t kDescKBitmap = 5;
static constexpr uint32_t kDescVBitmap = 6;
static constexpr uint32_t kInputTileElements = kTile * kHeadDimension;
static constexpr uint32_t kBitmapWords = kInputTileElements / 32;
static constexpr uint32_t kStageBytes = kLocalMemoryBytes / 2;

static inline float sparse_exp2(float x) {
  return 1.0f + x * (1.0f + 0.5f * x);
}

#ifdef FA_S_ENABLE_TIMING
static inline uint64_t timing_now() {
  return vx_rdcycle();
}
#endif

#undef __local_mem
#define __local_mem(size) (void *)(csr_read(VX_CSR_CTA_LMEM_ADDR))

static inline __attribute__((always_inline)) void launch_mma(
    uint32_t *a, uint32_t *b, float *c, float *d, uint32_t done,
    uint32_t *a_bitmap, uint32_t *b_bitmap, uint32_t max_a_blocks,
    uint32_t max_b_blocks, uint32_t sparsity) {
  const uintptr_t rs1 = (uintptr_t)vx_wgather(
      (size_t)(uintptr_t)a, (size_t)(uintptr_t)b,
      (size_t)(uintptr_t)c, (size_t)(uintptr_t)d);
  const uintptr_t rs2 = (uintptr_t)vx_wgather(
      (size_t)(uintptr_t)a_bitmap, (size_t)(uintptr_t)b_bitmap,
      (size_t)done,
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
  const bool __UNIFORM__ issuer = csr_read(VX_CSR_CTA_RANK) == 0;
#ifdef FA_S_ENABLE_TIMING
  auto timing = reinterpret_cast<uint64_t *>(arg->timing_addr) +
                tile_id * kTimingCount;
  const uint64_t total_start = timing_now();
  uint64_t phase_start = total_start;
  uint64_t qk_cycles;
  uint64_t softmax_cycles;
  uint64_t pv_cycles;
#endif

  auto scores = reinterpret_cast<float *>(arg->scores_addr) +
                tile_id * kAttentionGroups * kTile * kAttentionGroupSize;
  auto probabilities = reinterpret_cast<uint16_t *>(arg->probabilities_addr) +
                       tile_id * kAttentionGroups * kTile *
                           kAttentionGroupSize;
  auto partials = reinterpret_cast<float *>(arg->pv_addr) +
                  tile_id * kAttentionGroups * kTile * kHeadDimension;
  auto output = reinterpret_cast<float *>(arg->output_addr) +
                tile_id * kTile * kHeadDimension;

  auto lmem = reinterpret_cast<uint32_t *>(__local_mem(kLocalMemoryBytes));
  uint32_t *a_bitmap[2] = {
      lmem, lmem + kStageBytes / sizeof(uint32_t)};
  uint32_t *a_data[2] = {
      a_bitmap[0] + kBitmapWords, a_bitmap[1] + kBitmapWords};
  uint32_t *b_bitmap[2] = {
      lmem + kInputTileElements / 2 + 16,
      lmem + kStageBytes / sizeof(uint32_t) +
          kInputTileElements / 2 + 16};
  uint32_t *b_data[2] = {
      b_bitmap[0] + kBitmapWords + 16,
      b_bitmap[1] + kBitmapWords + 16};
  const uint32_t load[2] = {barrier_base, barrier_base + (1u << 8)};
  const uint32_t done[2] = {
      barrier_base + (2u << 8), barrier_base + (3u << 8)};

  for (uint32_t group = 0; group < kAttentionGroups; ++group) {
    const uint32_t key_tile = group * 2;
    float *pair_scores = scores + group * kTile * kAttentionGroupSize;
    vx_barrier(done[0], 1);
    if (issuer) {
      vx_dxa_issue_2d_wg(kDescQBitmap, load[0], a_bitmap[0], 0, tile_id);
      vx_dxa_issue_2d_wg(kDescKBitmap, load[0], b_bitmap[0], 0, key_tile);
      vx_dxa_issue_2d_wg(kDescQ, load[0], a_data[0], 0, tile_id);
      vx_dxa_issue_2d_wg(kDescK, load[0], b_data[0], 0, key_tile);
    }
    vx_barrier(load[0], 1);
    if (issuer)
      launch_mma(a_data[0], b_data[0], nullptr,
                 pair_scores, done[0],
                 a_bitmap[0], b_bitmap[0], arg->max_q_blocks,
                 arg->max_k_blocks, 2);

    vx_barrier(done[1], 1);
    if (issuer) {
      vx_dxa_issue_2d_wg(kDescQBitmap, load[1], a_bitmap[1], 0, tile_id);
      vx_dxa_issue_2d_wg(kDescKBitmap, load[1], b_bitmap[1], 0,
                         key_tile + 1);
      vx_dxa_issue_2d_wg(kDescQ, load[1], a_data[1], 0, tile_id);
      vx_dxa_issue_2d_wg(kDescK, load[1], b_data[1], 0, key_tile + 1);
    }
    vx_barrier(done[0], 1);
    vx_barrier(load[1], 1);
    if (issuer)
      launch_mma(a_data[1], b_data[1], nullptr,
                 pair_scores + kTileElements,
                 done[1], a_bitmap[1], b_bitmap[1], arg->max_q_blocks,
                 arg->max_k_blocks, 2);
    vx_barrier(done[1], 1);
  }

#ifdef FA_S_ENABLE_TIMING
  uint64_t phase_end = timing_now();
  qk_cycles = phase_end - phase_start;
  phase_start = phase_end;
#endif

  float row_max = -3.402823466e+38f;
  if (row < kTile) {
    for (uint32_t group = 0; group < kAttentionGroups; ++group)
      for (uint32_t tile = 0; tile < kAttentionGroupSize / kTile; ++tile) {
        const float *row_scores = scores +
            group * kTile * kAttentionGroupSize + tile * kTileElements +
            row * kTile;
        for (uint32_t key = 0; key < kTile; ++key) {
          const float score = row_scores[key] * 0.125f;
          row_max = score > row_max ? score : row_max;
        }
      }
  }

  float row_sum = 0.0f;
  if (row < kTile) {
    for (uint32_t group = 0; group < kAttentionGroups; ++group)
      for (uint32_t tile = 0; tile < kAttentionGroupSize / kTile; ++tile) {
        const float *row_scores = scores +
            group * kTile * kAttentionGroupSize + tile * kTileElements +
            row * kTile;
        uint16_t *column_probabilities = probabilities +
            group * kTile * kAttentionGroupSize + tile * kTile * kTile + row;
        for (uint32_t key = 0; key < kTile; ++key) {
          const float probability =
              sparse_exp2(row_scores[key] * 0.125f - row_max);
          column_probabilities[key * kTile] = float_to_fp16(probability);
          row_sum += probability;
        }
      }
  }
  __syncthreads();

#ifdef FA_S_ENABLE_TIMING
  phase_end = timing_now();
  softmax_cycles = phase_end - phase_start;
  phase_start = phase_end;
#endif

  for (uint32_t group = 0; group < kAttentionGroups; ++group) {
    uint32_t *p_data[2] = {lmem, lmem + kStageBytes / sizeof(uint32_t)};
    uint32_t *v_bitmap[2] = {
        p_data[0] + kInputTileElements / 2,
        p_data[1] + kInputTileElements / 2};
    uint32_t *v_data[2] = {
        v_bitmap[0] + kBitmapWords, v_bitmap[1] + kBitmapWords};

    const uint32_t output_tile = group * kOutputTiles;
    float *pair_partials = partials + group * kTile * kHeadDimension;
    vx_barrier(done[0], 1);
    if (issuer) {
      vx_dxa_issue_2d_wg(kDescP, load[0], p_data[0], 0,
                         tile_id * kAttentionGroups + group);
      vx_dxa_issue_2d_wg(kDescVBitmap, load[0], v_bitmap[0], 0,
                         output_tile);
      vx_dxa_issue_2d_wg(kDescV, load[0], v_data[0], 0,
                         output_tile);
    }
    vx_barrier(load[0], 1);
    if (issuer)
      launch_mma(p_data[0], v_data[0],
                 nullptr, pair_partials,
                 done[0], nullptr,
                 v_bitmap[0], 0, arg->max_v_blocks, 1);

    vx_barrier(done[1], 1);
    if (issuer) {
      vx_dxa_issue_2d_wg(kDescP, load[1], p_data[1], 0,
                         tile_id * kAttentionGroups + group);
      vx_dxa_issue_2d_wg(kDescVBitmap, load[1], v_bitmap[1], 0,
                         output_tile + 1);
      vx_dxa_issue_2d_wg(kDescV, load[1], v_data[1], 0,
                         output_tile + 1);
    }
    vx_barrier(done[0], 1);
    vx_barrier(load[1], 1);
    if (issuer)
      launch_mma(p_data[1], v_data[1],
                 nullptr,
                 pair_partials + kTileElements,
                 done[1], nullptr, v_bitmap[1], 0,
                 arg->max_v_blocks, 1);
    vx_barrier(done[1], 1);
  }

#ifdef FA_S_ENABLE_TIMING
  phase_end = timing_now();
  pv_cycles = phase_end - phase_start;
  phase_start = phase_end;
#endif

  if (row < kTile) {
    const float inverse_sum = 1.0f / row_sum;
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      float value = 0.0f;
      for (uint32_t group = 0; group < kAttentionGroups; ++group) {
        const uint32_t index =
            group * kTile * kHeadDimension +
            (column / kTile) * kTileElements + row * kTile + column % kTile;
        value += partials[index];
      }
      output[row * kHeadDimension + column] = value * inverse_sum;
    }
  }
#ifdef FA_S_ENABLE_TIMING
  const uint64_t end = timing_now();
  if (row == 0) {
    timing[kTimingQkMma] = qk_cycles;
    timing[kTimingSoftmax] = softmax_cycles;
    timing[kTimingPvMma] = pv_cycles;
    timing[kTimingFinal] = end - phase_start;
    timing[kTimingTotal] = end - total_start;
  }
#endif
}
