#pragma once

#include <cmath>

#include "pinn/core/state.hpp"
#include "pinn/core/types.hpp"

namespace systems {
namespace ode {
struct ElasticPendulum {
  using state_t = core::State<4>;
  using data_t = state_t::data_t;

  static constexpr core::index_t dof = state_t::dof;

  constexpr ElasticPendulum(data_t mass, data_t gravity, data_t stiffness,
                            data_t length) noexcept
      : mass_{mass},
        gravity_{gravity},
        stiffness_{stiffness},
        length_{length} {}

  state_t dudt(data_t time, const state_t& state) const {
    const data_t pth_over_m_r_r{state[3] / mass_ / state[0] / state[0]};
    return state_t{state[2] / mass_, pth_over_m_r_r,
                   pth_over_m_r_r * state[3] / state[0] -
                       stiffness_ * (state[0] - length_) +
                       mass_ * gravity_ * std::cos(state[1]),
                   -mass_ * gravity_ * state[0] * std::sin(state[1])};
  }

  data_t hamiltonian(const state_t& state) const {
    return 0.5 * ((state[2] * state[2] +
                   state[3] * state[3] / state[0] / state[0]) /
                      mass_ +
                  stiffness_ * (state[0] - length_) * (state[0] - length_)) -
           mass_ * gravity_ * state[0] * std::cos(state[1]);
  }

 private:
  data_t mass_;
  data_t gravity_;
  data_t stiffness_;
  data_t length_;
};
}  // namespace ode
}  // namespace systems
