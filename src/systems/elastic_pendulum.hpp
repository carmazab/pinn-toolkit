#pragma once

#include <cmath>

#include "core/state.hpp"
#include "core/types.hpp"

namespace systems {
struct ElasticPendulum {
  using state_t = core::State<4>;

  static constexpr core::index_t dof = state_t::dof;

  constexpr ElasticPendulum(double mass, double gravity, double stiffness,
                            double length) noexcept
      : mass_{mass},
        gravity_{gravity},
        stiffness_{stiffness},
        length_{length} {}

  state_t dudt(double t, const state_t& state) const {
    const double pth_over_m_r_r{state[3] / mass_ / state[0] / state[0]};
    return state_t{state[2] / mass_, pth_over_m_r_r,
                   pth_over_m_r_r * state[3] / state[0] -
                       stiffness_ * (state[0] - length_) +
                       mass_ * gravity_ * std::cos(state[1]),
                   -mass_ * gravity_ * state[0] * std::sin(state[1])};
  }

  double hamiltonian(const state_t& state) const {
    return 0.5 * ((state[2] * state[2] +
                   state[3] * state[3] / state[0] / state[0]) /
                      mass_ +
                  stiffness_ * (state[0] - length_) * (state[0] - length_)) -
           mass_ * gravity_ * state[0] * std::cos(state[1]);
  }

 private:
  double mass_;
  double gravity_;
  double stiffness_;
  double length_;
};
}  // namespace systems
