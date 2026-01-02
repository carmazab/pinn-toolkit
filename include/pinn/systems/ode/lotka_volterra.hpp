#pragma once

#include <cmath>

#include "pinn/core/state.hpp"
#include "pinn/core/types.hpp"

namespace systems {
namespace ode {
struct LotkaVolterra {
  using state_t = core::State<2>;
  using data_t = state_t::data_t;

  static constexpr core::index_t dof = state_t::dof;

  constexpr LotkaVolterra(data_t a, data_t b, data_t c, data_t d) noexcept
      : a_{a}, b_{b}, c_{c}, d_{d} {}

  constexpr state_t dudt(data_t time, const state_t& state) const noexcept {
    return state_t{state[0] * (a_ - b_ * state[1]),
                   state[1] * (d_ * state[0] - c_)};
  }

  data_t hamiltonian(const state_t& state) const {
    return d_ * state[0] - c_ * std::log(state[0]) + b_ * state[1] -
           a_ * std::log(state[1]);
  }

 private:
  data_t a_;
  data_t b_;
  data_t c_;
  data_t d_;
};
}  // namespace ode
}  // namespace systems
