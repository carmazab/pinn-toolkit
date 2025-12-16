#pragma once

#include "core/concepts.hpp"
#include "core/state.hpp"
#include "core/types.hpp"

namespace testhelpers {
struct DummySystem {
  static constexpr core::index_t dof = 4;

  using state_t = core::State<dof>;

  state_t dudt(double t, const state_t& state) const noexcept {
    return state_t{state[0] * state[3], 1.23 * state[2], -state[3] + state[1],
                   state[2] - 4.0 * state[1] * state[0]};
  }
};
}  // namespace testhelpers