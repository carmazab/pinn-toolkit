#include <string>

#include "core/types.hpp"
#include "framework/check_equal.hpp"
#include "framework/helpers.hpp"
#include "numerics/timesteppers/runge_kutta_4.hpp"

namespace {
void run_test_suite() {
  using system_t = testing::helpers::DummySystem;
  using state_t = system_t::state_t;
  using data_t = system_t::data_t;
  system_t system;

  using stepper_t = numerics::timesteppers::RungeKutta4<system_t>;

  const data_t step_size{0.014};
  stepper_t stepper{step_size};

  const data_t time{0.97};
  state_t state{1.23, 3.21, 4.31, 5.13};

  const data_t half_step{0.5 * step_size};
  const state_t k1{system.dudt(time, state)};
  const state_t k2{system.dudt(time + half_step, state + half_step * k1)};
  const state_t k3{system.dudt(time + half_step, state + half_step * k2)};
  const state_t k4{system.dudt(time + step_size, state + step_size * k3)};

  state_t expected{state};
  for (core::index_t j{0}; j < system_t::dof; ++j) {
    expected[j] += step_size * (k1[j] + 2.0 * (k2[j] + k3[j]) + k4[j]) / 6.0;
  }

  testing::check_equal_within(stepper.update(time, state, system), expected);

  const stepper_t::Parameters params{step_size};
  testing::check_equal(params.step_size.name(), std::string{"step_size"});
  testing::check_equal_within(params.step_size.value, step_size);

  stepper_t from_params{params};
  testing::check_equal_within(from_params.step_size(), step_size);
}
}  // namespace

TEST_CASE("unit.numerics.timesteppers.runge_kutta_4",
          "[unit][numerics][timesteppers]") {
  run_test_suite();
}