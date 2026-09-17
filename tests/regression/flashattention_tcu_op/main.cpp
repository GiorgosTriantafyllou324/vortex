#include "common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <vortex.h>
#include <dxa.h>

#define RT_CHECK(expr)                                                   \
  do {                                                                   \
    const int status = (expr);                                           \
    if (status != 0) {                                                   \
      std::cerr << "Error: " << #expr << " returned " << status << '\n'; \
      cleanup();                                                         \
      return 1;                                                          \
    }                                                                    \
  } while (false)

static constexpr float kTolerance = 3.0e-3f;

vx_device_h device = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
vx_buffer_h q_buffer = nullptr;
vx_buffer_h k_buffer = nullptr;
vx_buffer_h v_buffer = nullptr;
vx_buffer_h scores_buffer = nullptr;
vx_buffer_h probabilities_buffer = nullptr;
vx_buffer_h pv_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h q_bitmap_buffer = nullptr;
vx_buffer_h k_bitmap_buffer = nullptr;
vx_buffer_h v_bitmap_buffer = nullptr;
#ifdef FA_S_ENABLE_TIMING
vx_buffer_h timing_buffer = nullptr;
#endif

static void cleanup() {
  if (q_buffer) vx_mem_free(q_buffer);
  if (k_buffer) vx_mem_free(k_buffer);
  if (v_buffer) vx_mem_free(v_buffer);
  if (scores_buffer) vx_mem_free(scores_buffer);
  if (probabilities_buffer) vx_mem_free(probabilities_buffer);
  if (pv_buffer) vx_mem_free(pv_buffer);
  if (output_buffer) vx_mem_free(output_buffer);
  if (q_bitmap_buffer) vx_mem_free(q_bitmap_buffer);
  if (k_bitmap_buffer) vx_mem_free(k_bitmap_buffer);
  if (v_bitmap_buffer) vx_mem_free(v_bitmap_buffer);
#ifdef FA_S_ENABLE_TIMING
  if (timing_buffer) vx_mem_free(timing_buffer);
#endif
  if (krnl_buffer) vx_mem_free(krnl_buffer);
  if (args_buffer) vx_mem_free(args_buffer);
  if (device) {
    vx_dump_perf(device, stdout);
    vx_dev_close(device);
  }
}
static float input_value(uint32_t row, uint32_t column, uint32_t salt) {
  const int32_t value =
      static_cast<int32_t>((row * 7 + column * 3 + salt) % 17) - 8;
  return static_cast<float>(value) * 0.0625f;
}

static std::vector<uint16_t> pack_a(const std::vector<float>& source,
                                    uint32_t rows, uint32_t columns) {
  std::vector<uint16_t> packed(source.size());
  uint32_t tile_id = 0;
  for (uint32_t tile_row = 0; tile_row < rows; tile_row += kTile) {
    for (uint32_t tile_col = 0; tile_col < columns;
         tile_col += kTile, ++tile_id) {
      const uint32_t tile_base = tile_id * kTileElements;
      for (uint32_t column = 0; column < kTile; ++column) {
        for (uint32_t row = 0; row < kTile; ++row) {
          packed[tile_base + column * kTile + row] = float_to_fp16(
              source[(tile_row + row) * columns + tile_col + column]);
        }
      }
    }
  }
  return packed;
}

static std::vector<uint16_t> pack_k_transposed(
    const std::vector<float>& source) {
  std::vector<uint16_t> packed(source.size());
  for (uint32_t group = 0; group < kKeyTiles; ++group) {
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
      for (uint32_t key = 0; key < kTile; ++key) {
        packed[group * kTile * kHeadDimension + dimension * kTile + key] =
            float_to_fp16(
            source[(group * kTile + key) * kHeadDimension + dimension]);
      }
    }
  }
  return packed;
}

static std::vector<uint16_t> pack_v(const std::vector<float>& source) {
  std::vector<uint16_t> packed(source.size());
  for (uint32_t group = 0; group < kAttentionGroups; ++group) {
    for (uint32_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
      const uint32_t tile_base =
          (group * kOutputTiles + output_tile) * kAttentionGroupSize * kTile;
      for (uint32_t key = 0; key < kAttentionGroupSize; ++key) {
        for (uint32_t column = 0; column < kTile; ++column) {
          packed[tile_base + key * kTile + column] = float_to_fp16(
              source[(group * kAttentionGroupSize + key) * kHeadDimension +
                     output_tile * kTile + column]);
        }
      }
    }
  }
  return packed;
}

#ifndef FA_SPARSITY
#define FA_SPARSITY 0
#endif

#ifndef FA_ZERO_PERCENT
#define FA_ZERO_PERCENT 50
#endif

struct sparse_tiles_t {
  std::vector<uint16_t> values;
  std::vector<uint8_t> bitmap;
  uint32_t max_blocks;
};

static sparse_tiles_t compress_tiles(const std::vector<uint16_t>& dense) {
  const uint32_t tile_elements = kTile * kHeadDimension;
  const uint32_t tile_count = dense.size() / tile_elements;
  sparse_tiles_t result = {
      std::vector<uint16_t>(dense.size(), 0),
      std::vector<uint8_t>(dense.size() / 8, 0), 0};
  uint32_t max_nonzeros = 0;
  for (uint32_t tile = 0; tile < tile_count; ++tile) {
    uint32_t nonzeros = 0;
    for (uint32_t index = 0; index < tile_elements; ++index) {
      const uint16_t value = dense[tile * tile_elements + index];
      if (value != 0) {
        result.bitmap[(tile * tile_elements + index) / 8] |=
            static_cast<uint8_t>(1u << (index & 7u));
        result.values[tile * tile_elements + nonzeros++] = value;
      }
    }
    max_nonzeros = nonzeros > max_nonzeros ? nonzeros : max_nonzeros;
  }
  result.max_blocks = (max_nonzeros * sizeof(uint16_t) + 127) / 128;
  return result;
}

static bool prune_element(uint32_t row, uint32_t column, uint32_t salt) {
  return ((row * 37u + column * 19u + salt * 13u) % 100u) <
         FA_ZERO_PERCENT;
}

static int create_buffer(size_t bytes, uint32_t flags, vx_buffer_h *buffer,
                         uint64_t *address) {
  int status = vx_mem_alloc(device, bytes, flags, buffer);
  return status != 0 ? status : vx_mem_address(*buffer, address);
}

int main() {
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0 ||
      (isa_flags & VX_ISA_EXT_DXA) == 0) {
    std::cerr << "TCU_OP requires both TCU and DXA extensions\n";
    cleanup();
    return 1;
  }

  const size_t q_elements = kTotalQueryRows * kHeadDimension;
  const size_t kv_elements = kSequenceLength * kHeadDimension;
  const size_t intermediate_elements =
      kQueryTiles * kKeyTiles * kTileElements;
  const size_t output_elements = kTotalQueryRows * kHeadDimension;

  std::vector<float> q_values(q_elements);
  std::vector<float> k_values(kv_elements);
  std::vector<float> v_values(kv_elements);
  for (uint32_t row = 0; row < kTotalQueryRows; ++row) {
    for (uint32_t column = 0; column < kHeadDimension; ++column)
      q_values[row * kHeadDimension + column] =
          (FA_SPARSITY == 2 && prune_element(row, column, 1))
              ? 0.0f
              : input_value(row, column, 1);
  }
  for (uint32_t row = 0; row < kSequenceLength; ++row) {
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      k_values[row * kHeadDimension + column] =
          (FA_SPARSITY == 2 && prune_element(row, column, 5))
              ? 0.0f
              : input_value(row, column, 5);
      v_values[row * kHeadDimension + column] =
          (FA_SPARSITY == 2 && prune_element(row, column, 11))
              ? 0.0f
              : input_value(row, column, 11);
    }
  }

  const auto dense_q = pack_a(q_values, kTotalQueryRows, kHeadDimension);
  const auto dense_k = pack_k_transposed(k_values);
  const auto dense_v = pack_v(v_values);
  const auto sparse_q = compress_tiles(dense_q);
  const auto sparse_k = compress_tiles(dense_k);
  const auto sparse_v = compress_tiles(dense_v);
  const auto& q = FA_SPARSITY == 2 ? sparse_q.values : dense_q;
  const auto& k = FA_SPARSITY == 2 ? sparse_k.values : dense_k;
  const auto& v = FA_SPARSITY == 2 ? sparse_v.values : dense_v;
  std::vector<float> output(output_elements, 0.0f);
  std::vector<float> reference(output_elements, 0.0f);
#ifdef FA_S_ENABLE_TIMING
  std::vector<uint64_t> timing(kQueryTiles * kTimingCount, 0);
#endif

  for (uint32_t row = 0; row < kTotalQueryRows; ++row) {
    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    for (uint32_t group = 0; group < kAttentionGroups; ++group) {
      float scores[kAttentionGroupSize] = {};
      float group_max = -3.402823466e+38f;
      for (uint32_t key = 0; key < kAttentionGroupSize; ++key) {
        const uint32_t global_key = group * kAttentionGroupSize + key;
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
          scores[key] +=
              q_values[row * kHeadDimension + dimension] *
              k_values[global_key * kHeadDimension + dimension];
        scores[key] *= 0.125f;
        group_max = scores[key] > group_max ? scores[key] : group_max;
      }
      const float new_max = running_max > group_max ? running_max : group_max;
      const float previous_scale = running_sum == 0.0f
                                       ? 0.0f
                                       : exp2_taylor(running_max - new_max);
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
        reference[row * kHeadDimension + dimension] *= previous_scale;
      float group_sum = 0.0f;
      for (uint32_t key = 0; key < kAttentionGroupSize; ++key) {
        const uint32_t global_key = group * kAttentionGroupSize + key;
        const float probability = exp2_taylor(scores[key] - new_max);
        group_sum += probability;
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
          reference[row * kHeadDimension + dimension] +=
              probability * v_values[global_key * kHeadDimension + dimension];
      }
      running_sum = running_sum * previous_scale + group_sum;
      running_max = new_max;
    }
    for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
      reference[row * kHeadDimension + dimension] /= running_sum;
  }

  kernel_arg_t args = {};
  args.max_q_blocks = sparse_q.max_blocks;
  args.max_k_blocks = sparse_k.max_blocks;
  args.max_v_blocks = sparse_v.max_blocks;
  RT_CHECK(create_buffer(q.size() * sizeof(uint16_t), VX_MEM_READ,
                         &q_buffer, &args.q_addr));
  RT_CHECK(create_buffer(k.size() * sizeof(uint16_t), VX_MEM_READ,
                         &k_buffer, &args.k_addr));
  RT_CHECK(create_buffer(v.size() * sizeof(uint16_t), VX_MEM_READ,
                         &v_buffer, &args.v_addr));
  RT_CHECK(create_buffer(intermediate_elements * sizeof(float),
                         VX_MEM_READ_WRITE, &scores_buffer,
                         &args.scores_addr));
  RT_CHECK(create_buffer(kQueryTiles * kTile * kSequenceLength *
                             sizeof(uint16_t),
                         VX_MEM_READ_WRITE, &probabilities_buffer,
                         &args.probabilities_addr));
  RT_CHECK(create_buffer(intermediate_elements * sizeof(float),
                         VX_MEM_READ_WRITE, &pv_buffer, &args.pv_addr));
  RT_CHECK(create_buffer(output.size() * sizeof(float), VX_MEM_READ_WRITE,
                         &output_buffer, &args.output_addr));
#ifdef FA_S_ENABLE_TIMING
  RT_CHECK(create_buffer(timing.size() * sizeof(uint64_t), VX_MEM_WRITE,
                         &timing_buffer, &args.timing_addr));
#endif
#if FA_SPARSITY == 2
  RT_CHECK(create_buffer(sparse_q.bitmap.size(), VX_MEM_READ,
                         &q_bitmap_buffer, &args.q_bitmap_addr));
  RT_CHECK(create_buffer(sparse_k.bitmap.size(), VX_MEM_READ,
                         &k_bitmap_buffer, &args.k_bitmap_addr));
  RT_CHECK(create_buffer(sparse_v.bitmap.size(), VX_MEM_READ,
                         &v_bitmap_buffer, &args.v_bitmap_addr));
#endif

  RT_CHECK(vx_copy_to_dev(q_buffer, q.data(), 0, q.size() * sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(k_buffer, k.data(), 0, k.size() * sizeof(uint16_t)));
  RT_CHECK(vx_copy_to_dev(v_buffer, v.data(), 0, v.size() * sizeof(uint16_t)));
#if FA_SPARSITY == 2
  RT_CHECK(vx_copy_to_dev(q_bitmap_buffer, sparse_q.bitmap.data(), 0,
                          sparse_q.bitmap.size()));
  RT_CHECK(vx_copy_to_dev(k_bitmap_buffer, sparse_k.bitmap.data(), 0,
                          sparse_k.bitmap.size()));
  RT_CHECK(vx_copy_to_dev(v_bitmap_buffer, sparse_v.bitmap.data(), 0,
                          sparse_v.bitmap.size()));
#endif

#if FA_SPARSITY == 2
  const uint32_t q_transfer = args.max_q_blocks * 128 / sizeof(uint16_t);
  const uint32_t k_transfer = args.max_k_blocks * 128 / sizeof(uint16_t);
  const uint32_t v_transfer = args.max_v_blocks * 128 / sizeof(uint16_t);
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 0, args.q_addr, q_transfer, kQueryTiles,
      kTile * kHeadDimension * sizeof(uint16_t), q_transfer, 1,
      sizeof(uint16_t)));
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 1, args.k_addr, k_transfer, kKeyTiles,
      kTile * kHeadDimension * sizeof(uint16_t), k_transfer, 1,
      sizeof(uint16_t)));
#else
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 0, args.q_addr, kTile * kHeadDimension, kQueryTiles,
      kTile * kHeadDimension * sizeof(uint16_t), kTile * kHeadDimension, 1,
      sizeof(uint16_t)));
#endif
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 1, args.k_addr, kTile * kHeadDimension, kKeyTiles,
      kTile * kHeadDimension * sizeof(uint16_t), kTile * kHeadDimension, 1,
      sizeof(uint16_t)));
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 2, args.probabilities_addr, kTile * kAttentionGroupSize,
      kQueryTiles * kAttentionGroups,
      kTile * kAttentionGroupSize * sizeof(uint16_t),
      kTile * kAttentionGroupSize, 1, sizeof(uint16_t)));
#if FA_SPARSITY == 2
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 3, args.v_addr, v_transfer, kAttentionGroups * kOutputTiles,
      kAttentionGroupSize * kTile * sizeof(uint16_t), v_transfer, 1,
      sizeof(uint16_t)));
  const uint32_t bitmap_words = kTile * kHeadDimension / 32;
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 4, args.q_bitmap_addr, bitmap_words, kQueryTiles,
      bitmap_words * sizeof(uint32_t), bitmap_words, 1, sizeof(uint32_t)));
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 5, args.k_bitmap_addr, bitmap_words, kKeyTiles,
      bitmap_words * sizeof(uint32_t), bitmap_words, 1, sizeof(uint32_t)));
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 6, args.v_bitmap_addr, bitmap_words,
      kAttentionGroups * kOutputTiles, bitmap_words * sizeof(uint32_t),
      bitmap_words, 1, sizeof(uint32_t)));
#else
  RT_CHECK(vx_dxa_program_desc_2d(
      device, 3, args.v_addr, kAttentionGroupSize * kTile,
      kAttentionGroups * kOutputTiles,
      kAttentionGroupSize * kTile * sizeof(uint16_t),
      kAttentionGroupSize * kTile, 1, sizeof(uint16_t)));
#endif

  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(args), &args_buffer));
  uint32_t grid_dim[1] = {kQueryTiles};
  uint32_t block_dim[1] = {NUM_THREADS};
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim,
                      kLocalMemoryBytes));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  RT_CHECK(vx_copy_from_dev(output.data(), output_buffer, 0,
                            output.size() * sizeof(float)));
#ifdef FA_S_ENABLE_TIMING
  RT_CHECK(vx_copy_from_dev(timing.data(), timing_buffer, 0,
                            timing.size() * sizeof(uint64_t)));
#endif

  uint32_t errors = 0;
  float max_error = 0.0f;
  for (size_t index = 0; index < output.size(); ++index) {
    const float error = std::fabs(output[index] - reference[index]);
    max_error = error > max_error ? error : max_error;
    if (error > kTolerance) {
      if (errors < 16)
        std::cerr << "Mismatch at " << index << ": expected "
                  << reference[index] << ", got " << output[index]
                  << ", error " << error << '\n';
      ++errors;
    }
  }

  cleanup();
  if (errors != 0) {
    std::cerr << "FAILED: " << errors << " mismatches, max error "
              << max_error << '\n';
    return 1;
  }
#ifdef FA_S_ENABLE_TIMING
  uint64_t totals[kTimingCount] = {};
  for (uint32_t tile = 0; tile < kQueryTiles; ++tile)
    for (uint32_t counter = 0; counter < kTimingCount; ++counter)
      totals[counter] += timing[tile * kTimingCount + counter];
  const char *names[kTimingCount] = {
      "total", "QK MMA (S=2)", "softmax", "PV MMA (S=1)", "final"};
  const uint64_t phases = totals[kTimingQkMma] + totals[kTimingSoftmax] +
                          totals[kTimingPvMma] + totals[kTimingFinal];
  std::cout << "Timing cycles across " << kQueryTiles << " query tiles:\n";
  for (uint32_t counter = 0; counter < kTimingCount; ++counter) {
    const uint64_t denominator = counter == kTimingTotal ? totals[kTimingTotal]
                                                         : phases;
    const double percent = denominator == 0 ? 0.0
        : 100.0 * static_cast<double>(totals[counter]) / denominator;
    std::cout << "  " << std::setw(15) << names[counter] << ": "
              << std::setw(10) << totals[counter] << " cycles ("
              << std::fixed << std::setprecision(1) << percent << "%)\n"
              << std::defaultfloat;
  }
#endif
  std::cout << "PASSED: FlashAttention TCU_OP forward, Q="
            << kTotalQueryRows << ", KV=" << kSequenceLength
            << ", D=" << kHeadDimension << ", max error=" << max_error
            << '\n';
  return 0;
}
