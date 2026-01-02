#pragma once

#include <cstdint>
#include <random>

#include "pinn/core/concepts.hpp"

namespace core {
namespace random {
struct RngState {
  using engine_t = std::mt19937_64;

  RngState(const RngState&) = default;
  RngState& operator=(const RngState&) = default;
  RngState(RngState&&) noexcept = default;
  RngState& operator=(RngState&&) noexcept = default;
  ~RngState() noexcept = default;

  explicit RngState(std::uint64_t seed) noexcept
      : seed_{seed},
        engine_{seed},
        standard_normal_{0.0, 1.0},
        standard_uniform_{0.0, 1.0} {}

  engine_t& engine() noexcept { return engine_; }
  const engine_t& engine() const noexcept { return engine_; }

  constexpr std::uint64_t seed() const noexcept { return seed_; }

  template <core::concepts::Real RealT>
  RealT normal(RealT mean, RealT stddev) {
    return static_cast<RealT>(mean + stddev * standard_normal_(engine_));
  }

  template <core::concepts::Real RealT>
  RealT uniform(RealT min, RealT max) {
    return static_cast<RealT>(min + (max - min) * standard_uniform_(engine_));
  }

  template <core::concepts::Integer IntT>
  IntT uniform_int(IntT min, IntT max) {
    return static_cast<IntT>(std::uniform_int_distribution{min, max}(engine_));
  }

  RngState split(std::uint64_t stream_id) const noexcept {
    return make_stream(seed_, stream_id);
  }

  static RngState make_stream(std::uint64_t master_seed,
                              std::uint64_t stream_id) noexcept {
    // blatant 64-bit-version copy of Boost's hash combine formula. See
    // https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine
    std::uint64_t mixed =
        master_seed ^ (stream_id + 0x9e3779b97f4a7c15ULL + (master_seed << 6) +
                       (master_seed >> 2));
    return RngState{mixed};
  }

 private:
  std::uint64_t seed_;
  engine_t engine_;
  std::normal_distribution<double> standard_normal_;
  std::uniform_real_distribution<double> standard_uniform_;
};
}  // namespace random
}  // namespace core