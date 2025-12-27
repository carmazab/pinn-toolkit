#pragma once

#include "core/state.hpp"
#include "core/types.hpp"

namespace testhelpers {
struct DummySystem {
  using state_t = core::State<4>;
  using data_t = state_t::data_t;

  static constexpr core::index_t dof = state_t::dof;

  constexpr state_t dudt(data_t time, const state_t& state) const noexcept {
    return state_t{state[0] * state[3], 1.23 * state[2], -state[3] + state[1],
                   state[2] - 4.0 * state[1] * state[0]};
  }
};
}  // namespace testhelpers