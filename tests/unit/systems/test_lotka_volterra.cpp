#include <cmath>

#include "framework/check_equal.hpp"
#include "systems/lotka_volterra.hpp"

namespace {
void run_test_suite() {
  using system_t = systems::LotkaVolterra;
  using state_t = system_t::state_t;

  const double a{0.51}, b{0.65}, c{6.41}, d{0.763};
  system_t system{a, b, c, d};

  const double unused_time{99.0};
  state_t state{4.31, 5.13};

  const state_t expected_dudt{a * state[0] - b * state[0] * state[1],
                              -c * state[1] + d * state[0] * state[1]};
  testing::check_equal_within(system.dudt(unused_time, state), expected_dudt);

  const double expected_hamiltonian{d * state[0] - c * std::log(state[0]) +
                                    b * state[1] - a * std::log(state[1])};
  testing::check_equal_within(system.hamiltonian(state), expected_hamiltonian);
}
}  // namespace

TEST_CASE("unit.systems.lotka_volterra", "[unit][systems]") {
  run_test_suite();
}