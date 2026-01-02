#include "pinn/numerics/timesteppers/runge_kutta_4.hpp"
#include "pinn/numerics/timesteppers/traits.hpp"
#include "testing/check_equal.hpp"
#include "testing/helpers.hpp"

namespace {
void run_test_suite() {
  using stepper_t =
      numerics::timesteppers::RungeKutta4<testing::helpers::DummySystem>;
  numerics::timesteppers::Traits<stepper_t> traits;

  testing::helpers::DummyNode stepper;
  stepper.children["name"].value = "runge_kutta_4";
  stepper.children["parameters"].children["step_size"].value = 0.123;

  const stepper_t::Parameters params{
      traits.read_parameters(stepper["parameters"])};
  testing::check_equal_within(params.step_size.value, 0.123);
}
}  // namespace

TEST_CASE("unit.numerics.timesteppers.traits",
          "[unit][numerics][timesteppers]") {
  run_test_suite();
}