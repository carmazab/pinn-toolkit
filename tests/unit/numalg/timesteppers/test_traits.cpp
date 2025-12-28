#include <string>

#include "framework/check_equal.hpp"
#include "numalg/testhelpers.hpp"
#include "numalg/timesteppers/runge_kutta_4.hpp"
#include "numalg/timesteppers/traits.hpp"

namespace {
void run_test_suite() {
  using stepper_t = numalg::timesteppers::RungeKutta4<testhelpers::DummySystem>;
  numalg::timesteppers::Traits<stepper_t> traits;

  testhelpers::DummyNode stepper;
  stepper.children["name"].value = "runge_kutta_4";
  stepper.children["parameters"].children["step_size"].value = 0.123;

  const stepper_t::Parameters params{
      traits.read_parameters(stepper["parameters"])};
  testing::check_equal_within(params.step_size.value, 0.123);
}
}  // namespace

TEST_CASE("unit.numalg.timesteppers.traits", "[unit][numalg][timesteppers]") {
  run_test_suite();
}