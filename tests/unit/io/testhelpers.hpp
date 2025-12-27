#pragma once

#include <array>
#include <vector>

#include "core/state.hpp"
#include "core/types.hpp"

namespace testhelpers {
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
}  // namespace testhelpers