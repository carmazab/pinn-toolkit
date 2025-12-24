#pragma once

namespace numalg {
namespace timesteppers {
struct RungeKutta4 {
  explicit RungeKutta4(double step_size) noexcept
      : step_size_{step_size},
        half_step_{0.5 * step_size},
        step_sixths_{step_size / 6.0} {};

  template <class System>
  System::state_t update(double time, const System::state_t& state,
                         const System& system) const {
    using state_t = System::state_t;

    const state_t k1{system.dudt(time, state)};
    const state_t k2{system.dudt(time + half_step_, state + half_step_ * k1)};
    const state_t k3{system.dudt(time + half_step_, state + half_step_ * k2)};
    const state_t k4{system.dudt(time + step_size_, state + step_size_ * k3)};

    return state + step_sixths_ * (k1 + 2.0 * (k2 + k3) + k4);
  }

 private:
  double step_size_;
  double half_step_;
  double step_sixths_;
};
}  // namespace timesteppers
}  // namespace numalg