#ifndef FLASHATTENTION_TCU_OP_COMMON_H
#define FLASHATTENTION_TCU_OP_COMMON_H

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif
#ifndef K_TOTAL_QUERY_ROWS
#define K_TOTAL_QUERY_ROWS 32
#endif

#ifndef K_SEQUENCE_LENGTH
#define K_SEQUENCE_LENGTH 64
#endif

static constexpr uint32_t kTile = 32;
static constexpr uint32_t kTotalQueryRows = K_TOTAL_QUERY_ROWS;
static constexpr uint32_t kSequenceLength = K_SEQUENCE_LENGTH;
static constexpr uint32_t kHeadDimension = 64;
static constexpr uint32_t kQueryTiles = kTotalQueryRows / kTile;
static constexpr uint32_t kKeyTiles = kSequenceLength / kTile;
static constexpr uint32_t kAttentionGroupSize = 64;
static constexpr uint32_t kAttentionGroups =
    kSequenceLength / kAttentionGroupSize;
static constexpr uint32_t kOutputTiles = kHeadDimension / kTile;
static constexpr uint32_t kTileElements = kTile * kTile;
static constexpr uint32_t kLocalMemoryBytes = 1u << 14;

static_assert(kTotalQueryRows % kTile == 0,
              "query rows must be a multiple of the MMA_OP tile");
static_assert(kSequenceLength % kTile == 0,
              "sequence length must be a multiple of the MMA_OP tile");
static_assert(kSequenceLength % kAttentionGroupSize == 0 &&
                  kHeadDimension == 64,
              "FP16 MMA_OP attention requires KV multiples of 64 and D=64");

#ifdef FA_S_ENABLE_TIMING
enum timing_counter_t {
  kTimingTotal, kTimingQkMma, kTimingSoftmax, kTimingPvMma,
  kTimingFinal, kTimingCount
};
#endif

typedef struct {
  uint64_t q_addr;
  uint64_t k_addr;
  uint64_t v_addr;
  uint64_t scores_addr;
  uint64_t probabilities_addr;
  uint64_t pv_addr;
  uint64_t output_addr;
  uint64_t q_bitmap_addr;
  uint64_t k_bitmap_addr;
  uint64_t v_bitmap_addr;
  uint32_t max_q_blocks;
  uint32_t max_k_blocks;
  uint32_t max_v_blocks;
#ifdef FA_S_ENABLE_TIMING
  uint64_t timing_addr;
#endif
} kernel_arg_t;

static inline float exp2_taylor(float x) {
  return 1.0f + x + 0.5f * x * x;
}

static inline uint16_t float_to_fp16(float value) {
  union {
    float f;
    uint32_t u;
  } bits = {value};
  const uint32_t sign = (bits.u >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits.u >> 23) & 0xffu) - 127 + 15;
  if (exponent <= 0)
    return static_cast<uint16_t>(sign);
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00u);
  uint32_t mantissa = bits.u & 0x7fffffu;
  mantissa += 0xfffu + ((mantissa >> 13) & 1u);
  if (mantissa & 0x800000u) {
    mantissa = 0;
    ++exponent;
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa >> 13) & 0x3ffu));
}

#endif
