#pragma once

#include <cmath>

#include "core/state.hpp"
#include "core/types.hpp"

namespace systems {
struct LotkaVolterra {
  using state_t = core::State<2>;

  static constexpr core::index_t dof = state_t::dof;

  constexpr LotkaVolterra(double a, double b, double c, double d) noexcept
      : a_{a}, b_{b}, c_{c}, d_{d} {}

  constexpr state_t dudt(double t, const state_t& state) const noexcept {
    return state_t{state[0] * (a_ - b_ * state[1]),
                   state[1] * (d_ * state[0] - c_)};
  }

  double hamiltonian(const state_t& state) const {
    return d_ * state[0] - c_ * std::log(state[0]) + b_ * state[1] -
           a_ * std::log(state[1]);
  }

 private:
  double a_;
  double b_;
  double c_;
  double d_;
};
}  // namespace systems
