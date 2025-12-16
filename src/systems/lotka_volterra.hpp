#pragma once

#include <cmath>
#include <vector>

#include "core/state.hpp"
#include "core/types.hpp"

namespace systems {
struct LotkaVolterraState : public core::State<2> {
  constexpr LotkaVolterraState(double x, double y) noexcept
      : core::State<2>{x, y} {}

  constexpr double x() const noexcept { return data_[0]; }
  constexpr double y() const noexcept { return data_[1]; }
};

struct LotkaVolterra {
  using state_t = LotkaVolterraState;

  static constexpr std::size_t dof = 2;

  constexpr LotkaVolterra(double a, double b, double c, double d) noexcept
      : a_{a}, b_{b}, c_{c}, d_{d} {}

  constexpr state_t dudt(double t, const state_t& state) const noexcept {
    return state_t{state.x() * (a_ - b_ * state.y()),
                   state.y() * (d_ * state.x() - c_)};
  }

  double hamiltonian(const state_t& state) const {
    return d_ * state.x() - c_ * std::log(state.x()) + b_ * state.y() -
           a_ * std::log(state.y());
  }

 private:
  double a_;
  double b_;
  double c_;
  double d_;
};
}  // namespace systems
