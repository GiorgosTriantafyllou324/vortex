#include "common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_tensor.h>

#include <VX_config.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <stdint.h>

#ifndef SGEMM_CONST_M
#define SGEMM_CONST_M 32
#endif

#ifndef SGEMM_CONST_N
#define SGEMM_CONST_N 32
#endif

#ifndef SGEMM_CONST_K
#define SGEMM_CONST_K 32
#endif

#ifndef SGEMM_CONST_SPARSITY
#define SGEMM_CONST_SPARSITY 0
#endif

#define MARKER 0x12345678
#define BLUE_MARKER 0x12345677
#define GREEN_MARKER 0x12345676
#define PRE_TCU_MARKER 0x12345679

#ifndef SGEMM_TRACE_MARKERS
#define SGEMM_TRACE_MARKERS 1
#endif

#ifndef SGEMM_TRACE_STAGE_PLAN_MARKERS
#define SGEMM_TRACE_STAGE_PLAN_MARKERS 0
#endif

#if SGEMM_TRACE_MARKERS
#define TRACE_STORE(ptr, value) (*(ptr) = (value))
#else
#define TRACE_STORE(ptr, value) ((void)0)
#endif

#if SGEMM_TRACE_STAGE_PLAN_MARKERS
#define TRACE_STAGE_PLAN_STORE(ptr, value) TRACE_STORE(ptr, value)
#else
#define TRACE_STAGE_PLAN_STORE(ptr, value) ((void)0)
#endif

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
static constexpr uint32_t kDescA = 0;
static constexpr uint32_t kDescB = 1;
static constexpr uint32_t kDescC = 2;
static constexpr uint32_t kDescABitmap = 3;
static constexpr uint32_t kDescBBitmap = 4;

#undef __local_mem
#define __local_mem(size) \
  (void*)(csr_read(VX_CSR_CTA_LMEM_ADDR))

static constexpr uint32_t div_up_constexpr(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}



static inline __attribute__((always_inline)) uint64_t read_instret() {
#if __riscv_xlen == 64
  return csr_read(VX_CSR_MINSTRET);
#else
  uint32_t hi0, lo, hi1;
  do {
    hi0 = csr_read(VX_CSR_MINSTRET_H);
    lo  = csr_read(VX_CSR_MINSTRET);
    hi1 = csr_read(VX_CSR_MINSTRET_H);
  } while (hi0 != hi1);
  return (static_cast<uint64_t>(hi0) << 32) | lo;
#endif
}

__kernel void kernel_main(kernel_arg_t *__UNIFORM__ arg)
{
  const uint64_t instret_begin = read_instret();
  const __rdcycle_time cycle_begin = vx_rdcycle_sync_begin();

  auto pA = reinterpret_cast<uint32_t *>(arg->A_addr);
  auto pB = reinterpret_cast<uint32_t *>(arg->B_addr);
  auto pC = reinterpret_cast<uint32_t *>(arg->C_addr);
  auto pD = reinterpret_cast<uint32_t *>(arg->D_addr);
  auto pA_bitmap = reinterpret_cast<const uint32_t *>(arg->A_bitmap_addr);
  auto pB_bitmap = reinterpret_cast<const uint32_t *>(arg->B_bitmap_addr);

  const uint32_t max_a_blocks = arg->max_a_blocks;
  const uint32_t max_b_blocks = arg->max_b_blocks;
  static constexpr uint32_t M = SGEMM_CONST_M;
  static constexpr uint32_t N = SGEMM_CONST_N;
  static constexpr uint32_t K = SGEMM_CONST_K;
  const uint8_t sparsity = arg->sparsity;
  static constexpr uint8_t kConstSparsity = SGEMM_CONST_SPARSITY;
  static constexpr bool kDense = (kConstSparsity == 0);
  static constexpr bool kSparseA = (kConstSparsity == 2);
  static constexpr bool kSparseB = (kConstSparsity >= 1);

  static constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(ctx::input_t);
  static constexpr uint32_t o_ratio = sizeof(uint32_t) / sizeof(ctx::output_t);

  static constexpr uint32_t tile_M = 32;
  static constexpr uint32_t tile_N = 32;
  static constexpr uint32_t tile_K = 16 * i_ratio * 2;
  static constexpr uint32_t tiles_n = (N / tile_N);
  static constexpr uint32_t tiles_m = (M / tile_M);
  static constexpr uint32_t total_tiles = tiles_n * tiles_m;
  const uint32_t block_tile_id = blockIdx.y * gridDim.x + blockIdx.x;

  static constexpr uint32_t num_warps_per_cta = 1;
  // Stage-dependent IDs stay scalar to avoid spilling indexed barrier objects.
  const uint32_t barrier_base = get_local_group_id();

  static_assert (VX_CFG_LMEM_ENABLED);

  static constexpr uint32_t tileC_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t tileD_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t lmem_capacity_bytes = (1u << VX_CFG_LMEM_LOG_SIZE);
  static constexpr uint32_t half_lmem_bytes = lmem_capacity_bytes >> 1;
  static constexpr uint32_t dense_a_tile_regs = tile_M * tile_K / i_ratio;
  static constexpr uint32_t dense_b_tile_regs = tile_K * tile_N / i_ratio;

  uint32_t* lmem_base = reinterpret_cast<uint32_t *>(__local_mem(lmem_capacity_bytes));
  uint32_t* half0_base = lmem_base;
  uint32_t* half1_base = half0_base + (half_lmem_bytes / sizeof(uint32_t));
  static constexpr uint32_t half_lmem_regs = half_lmem_bytes / sizeof(uint32_t);

  const bool __UNIFORM__ is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  uint32_t current_stage = 0;
  uint32_t next_stage = 0;
  uintptr_t pending_rs1_val = 0;
  uintptr_t pending_rs2_val = 0;

  /* Lambda function is optimized by the compiler */
  auto launch_pending_mma = [&]() __attribute__((always_inline)) {
    vx_barrier(barrier_base + ((current_stage + 2) << 8), num_warps_per_cta);
    vx_barrier(barrier_base + (next_stage << 8), num_warps_per_cta);

    ctx::mma_op(pending_rs1_val, pending_rs2_val);

    current_stage = next_stage;
    next_stage ^= 1u;
  };


  if constexpr (kDense) 
  {
#pragma unroll
    for (uint32_t tile_row_idx = 0; tile_row_idx < tiles_m; ++tile_row_idx)
    {
#pragma unroll
      for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx)
      {

        const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
        const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
        uint32_t* mma_D = reinterpret_cast<uint32_t*>(mma_D_addr);

        static constexpr uint32_t kDenseLaunches = div_up_constexpr(K, tile_K);
        for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) 
        {
          const uint32_t k_tile_idx = dense_iter;

          const uint32_t flags_chunk = (uint32_t(dense_iter == 0) << 1) | uint32_t(dense_iter == (kDenseLaunches - 1));

          vx_barrier(barrier_base + ((current_stage + 2) << 8), num_warps_per_cta);

          /* In the very first execution, no data are ready so skip the mma_op */
          if ((tile_row_idx | tile_col_idx | dense_iter) != 0) {
            launch_pending_mma();
          }

          uint32_t* A_lmem_next = half0_base + next_stage * half_lmem_regs;
          uint32_t* B_lmem_next = A_lmem_next + dense_a_tile_regs;

          if (is_dxa_warp) {
            const uint32_t load_bar_id = barrier_base + (next_stage << 8);
            vx_barrier_expect_tx(load_bar_id, 2);

            vx_dxa_issue_3d_wg(kDescA, load_bar_id, A_lmem_next, 0, k_tile_idx, tile_row_idx);
            vx_dxa_issue_3d_wg(kDescB, load_bar_id, B_lmem_next, 0, k_tile_idx, tile_col_idx);
          }

          if (is_dxa_warp) {
            pending_rs1_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)A_lmem_next,
                      (size_t)(uintptr_t)B_lmem_next,
                      (size_t)(uintptr_t)nullptr /* mma_C */,
                      (size_t)(uintptr_t)mma_D_addr);
                      
            pending_rs2_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)nullptr /*mma_A_bitmap*/,
                      (size_t)(uintptr_t)nullptr /*mma_B_bitmap*/,
                      (size_t)(barrier_base + ((next_stage + 2) << 8)),
                      (size_t)((tile_K << 24) /*| (a_blocks << 18) | (b_blocks << 12)*/ | (vt::OTYPE::id << 8) |
                               (vt::ITYPE::id << 4) | (kConstSparsity << 2) | flags_chunk));
          }
        }
      }
    }

    launch_pending_mma();
    vx_barrier(barrier_base + ((current_stage + 2) << 8), num_warps_per_cta);
  } 
  
  else {

    /* Constants for the sparse case only */
    static constexpr uint32_t bitmap_tile_regs = tile_K * tile_M / 32;
    constexpr uint32_t b_bitmap_skew_regs = 16;
    constexpr uint32_t b_tile_align_regs = 16;

// #pragma unroll
    for (uint32_t tile_id = 0; tile_id < total_tiles; ++tile_id) {
      const uint32_t tile_row_idx = tile_id / tiles_n;
      const uint32_t tile_col_idx = tile_id % tiles_n;
      const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);

      static constexpr uint32_t kDenseLaunches = div_up_constexpr(K, tile_K);
// #pragma unroll
      for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter)
      {
        const uint32_t flags_chunk =
            (uint32_t(dense_iter == 0) << 1) |
            uint32_t(dense_iter == (kDenseLaunches - 1));

        if ((tile_id | dense_iter) != 0) {
          launch_pending_mma();
        }

        const uint32_t k_offset = dense_iter * tile_K;
        uint32_t* stage_base = half0_base + next_stage * half_lmem_regs;
        uint32_t* A_bitmap_lmem_next = nullptr;
        uint32_t* A_lmem_next = stage_base;
        uint32_t* B_bitmap_lmem_next = stage_base + dense_a_tile_regs;
        uint32_t* B_lmem_next = B_bitmap_lmem_next + bitmap_tile_regs;

        if constexpr (kSparseA) {
          A_bitmap_lmem_next = stage_base;
          A_lmem_next = stage_base + bitmap_tile_regs;
          B_bitmap_lmem_next += b_bitmap_skew_regs;
          B_lmem_next = B_bitmap_lmem_next + bitmap_tile_regs + b_tile_align_regs;
        }

        if (is_dxa_warp) {
          if constexpr (kSparseA) {
            const uint32_t load_bar_id = barrier_base + (next_stage << 8);
            vx_barrier_expect_tx(load_bar_id, 4);
            vx_dxa_issue_1d_wg(kDescABitmap, load_bar_id, A_bitmap_lmem_next, tile_row_idx * K + k_offset);
            vx_dxa_issue_2d_wg(kDescA, load_bar_id, A_lmem_next, 0, tile_row_idx * kDenseLaunches + dense_iter);
          } else {
            const uint32_t load_bar_id = barrier_base + (next_stage << 8);
            vx_barrier_expect_tx(load_bar_id, 3);
            vx_dxa_issue_2d_wg(kDescA, load_bar_id, A_lmem_next, 0, tile_row_idx * kDenseLaunches + dense_iter);
          }

          const uint32_t load_bar_id = barrier_base + (next_stage << 8);
          vx_dxa_issue_1d_wg(kDescBBitmap, load_bar_id, B_bitmap_lmem_next, tile_col_idx * K + k_offset);
          vx_dxa_issue_2d_wg(kDescB, load_bar_id, B_lmem_next, 0, tile_col_idx * kDenseLaunches + dense_iter);
        }

        if (is_dxa_warp) {
          pending_rs1_val = (uintptr_t)vx_wgather(
                    (size_t)(uintptr_t)A_lmem_next,
                    (size_t)(uintptr_t)B_lmem_next,
                    (size_t)(uintptr_t)nullptr,
                    (size_t)(uintptr_t)mma_D_addr);
          pending_rs2_val = (uintptr_t)vx_wgather(
                    (size_t)(uintptr_t)A_bitmap_lmem_next,
                    (size_t)(uintptr_t)B_bitmap_lmem_next,
                    (size_t)(barrier_base + ((next_stage + 2) << 8)),
                    (size_t)((tile_K << 24) | (max_a_blocks << 18) |
                            (max_b_blocks << 12) | (vt::OTYPE::id << 8) |
                            (vt::ITYPE::id << 4) | (kConstSparsity << 2) |
                            flags_chunk));
        }
      }
    }

    launch_pending_mma();
    vx_barrier(barrier_base + ((current_stage + 2) << 8), num_warps_per_cta);
  }

  const __rdcycle_time cycle_end = vx_rdcycle_sync_end();
  const uint64_t instret_end = read_instret();
  const uint64_t total_cycles = vx_rdcycle_sync_diff(cycle_begin, cycle_end);
  const uint64_t total_instructions = instret_end - instret_begin;

  if (vx_thread_id() == 0)
  {
    uint64_t* metrics = reinterpret_cast<uint64_t*>(arg->metrics_addr);
    metrics[0] = total_cycles;
    metrics[1] = total_instructions;
  }
}
