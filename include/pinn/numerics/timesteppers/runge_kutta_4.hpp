#pragma once

#include <string>

#include "pinn/systems/concepts.hpp"

namespace numerics {
namespace timesteppers {
template <systems::concepts::System SystemT>
struct RungeKutta4 {
  using system_t = SystemT;
  using state_t = system_t::state_t;
  using data_t = system_t::data_t;

  struct Parameters {
    struct StepSize {
      using type = data_t;
      type value;
      static constexpr std::string name() { return "step_size"; }
    } step_size;
  };

  static constexpr std::string name() { return "runge_kutta_4"; }

  explicit RungeKutta4(const Parameters& params) noexcept
      : RungeKutta4{params.step_size.value} {}

  explicit RungeKutta4(data_t step_size) noexcept
      : step_size_{step_size},
        half_step_{0.5 * step_size},
        step_sixths_{step_size / 6.0} {};

  constexpr data_t step_size() const noexcept { return step_size_; }

  state_t update(data_t time, const state_t& state,
                 const system_t& system) const {
    const state_t k1{system.dudt(time, state)};
    const state_t k2{system.dudt(time + half_step_, state + half_step_ * k1)};
    const state_t k3{system.dudt(time + half_step_, state + half_step_ * k2)};
    const state_t k4{system.dudt(time + step_size_, state + step_size_ * k3)};

    return state + step_sixths_ * (k1 + 2.0 * (k2 + k3) + k4);
  }

 private:
  data_t step_size_;

  data_t half_step_;
  data_t step_sixths_;
};
}  // namespace timesteppers
}  // namespace numerics