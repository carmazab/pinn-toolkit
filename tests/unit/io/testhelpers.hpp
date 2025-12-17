#pragma once

#include <array>
#include <vector>

#include "core/state.hpp"
#include "core/types.hpp"

namespace testhelpers {
struct DummyWriter {
  using state_t = core::State<4>;

  std::vector<double> t;
  std::array<std::vector<double>, state_t::dof> history;

  void append(double time, const state_t& state) {
    t.push_back(time);
    for (core::index_t s{0}; s < 4; ++s) {
      history[s].push_back(state[s]);
    }
  }
};
}  // namespace testhelpers