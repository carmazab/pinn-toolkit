#pragma once

namespace numalg {
namespace timesteppers {
struct RungeKutta4 {
  explicit RungeKutta4(double dt) noexcept
      : dt_{dt}, half_dt_{0.5 * dt}, dt_sixths_{dt / 6.0} {};

  template <class System>
  System::state_t update(double t, const System::state_t& state,
                         const System& system) const noexcept {
    using state_t = System::state_t;

    const state_t k1{system.dudt(t, state)};
    const state_t k2{system.dudt(t + half_dt_, state + half_dt_ * k1)};
    const state_t k3{system.dudt(t + half_dt_, state + half_dt_ * k2)};
    const state_t k4{system.dudt(t + dt_, state + dt_ * k3)};

    return state + dt_sixths_ * (k1 + 2.0 * (k2 + k3) + k4);
  }

 private:
  double dt_;
  double half_dt_;
  double dt_sixths_;
};
}  // namespace timesteppers
}  // namespace numalg