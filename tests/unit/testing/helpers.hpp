#pragma once

#include <any>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "pinn/core/state.hpp"
#include "pinn/core/types.hpp"

namespace testing {
namespace helpers {
struct DummyNode {
  template <typename T>
  T as() const {
    return std::any_cast<T>(value);
  }

  DummyNode operator[](const std::string& key) const {
    return children.at(key);
  }

  std::unordered_map<std::string, DummyNode> children;
  std::any value;
};

struct DummySystem {
  using state_t = core::State<4>;
  using data_t = state_t::data_t;

  static constexpr core::index_t dof = state_t::dof;

  constexpr state_t dudt(data_t time, const state_t& state) const noexcept {
    return state_t{state[0] * state[3], 1.23 * state[2], -state[3] + state[1],
                   state[2] - 4.0 * state[1] * state[0]};
  }
};

struct DummyWriter {
  using state_t = core::State<4>;
  using data_t = state_t::data_t;

  static constexpr core::index_t dof = state_t::dof;

  std::vector<double> t;
  std::array<std::vector<double>, dof> history;

  void append(double time, const state_t& state) {
    t.push_back(time);
    for (core::index_t s{0}; s < dof; ++s) {
      history[s].push_back(state[s]);
    }
  }

  void close() const noexcept {}
};
}  // namespace helpers
}  // namespace testing