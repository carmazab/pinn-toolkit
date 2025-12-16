#include "core/types.hpp"
#include "framework/check_equal.hpp"
#include "numalg/testhelpers.hpp"
#include "numalg/timesteppers/runge_kutta_4.hpp"

namespace {
void run_test_suite() {
  const double dt{0.014};
  numalg::timesteppers::RungeKutta4 stepper{dt};

  using system_t = testhelpers::DummySystem;
  using state_t = system_t::state_t;
  system_t system;

  const double t{0.97};
  state_t state{1.23, 3.21, 4.31, 5.13};

  const state_t k1{system.dudt(t, state)};
  const state_t k2{system.dudt(t + 0.5 * dt, state + 0.5 * dt * k1)};
  const state_t k3{system.dudt(t + 0.5 * dt, state + 0.5 * dt * k2)};
  const state_t k4{system.dudt(t + dt, state + dt * k3)};

  state_t expected{state};
  for (core::index_t j{0}; j < system_t::dof; ++j) {
    expected[j] += dt * (k1[j] + 2.0 * (k2[j] + k3[j]) + k4[j]) / 6.0;
  }

  testing::check_equal_within(stepper.update(t, state, system), expected);
}
}  // namespace

TEST_CASE("unit.numalg.timesteppers.runge_kutta_4",
          "[unit][numalg][timesteppers]") {
  run_test_suite();
}