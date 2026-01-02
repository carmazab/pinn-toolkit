#include "pinn/core/random/rng_state.hpp"
#include "pinn/numerics/timesteppers/runge_kutta_4.hpp"
#include "pinn/numerics/timesteppers/traits.hpp"
#include "testing/check_equal.hpp"
#include "testing/helpers.hpp"
#include "testing/random_seed.hpp"

namespace {
void run_test_suite() {
  using stepper_t =
      numerics::timesteppers::RungeKutta4<testing::helpers::DummySystem>;
  numerics::timesteppers::Traits<stepper_t> traits;

  core::random::RngState rng{testing::random_seed()};
  const stepper_t::data_t step_size{rng.uniform(0.0, 3.0)};

  testing::helpers::DummyNode stepper;
  stepper.children["name"].value = "runge_kutta_4";
  stepper.children["parameters"].children["step_size"].value = step_size;

  const stepper_t::Parameters params{
      traits.read_parameters(stepper["parameters"])};
  testing::check_equal_within(params.step_size.value, step_size);
}
}  // namespace

TEST_CASE("unit.numerics.timesteppers.traits",
          "[unit][numerics][timesteppers]") {
  run_test_suite();
}