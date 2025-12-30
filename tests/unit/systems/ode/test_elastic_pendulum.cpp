#include <cmath>

#include "framework/check_equal.hpp"
#include "systems/ode/elastic_pendulum.hpp"

namespace {
void run_test_suite() {
  using system_t = systems::ode::ElasticPendulum;
  using state_t = system_t::state_t;

  const double m{1.51}, g{9.123}, k{0.531}, l{0.98};
  system_t system{m, g, k, l};

  const double unused_time{99.0};
  state_t state{1.24, 0.421, 0.15, -0.512};

  const state_t expected_dudt{
      state[2] / m, state[3] / m / state[0] / state[0],
      state[3] * state[3] / m / state[0] / state[0] / state[0] -
          k * (state[0] - l) + m * g * std::cos(state[1]),
      -m * g * state[0] * std::sin(state[1])};
  testing::check_equal_within(system.dudt(unused_time, state), expected_dudt);

  const double expected_hamiltonian{state[2] * state[2] / 2.0 / m +
                                    state[3] * state[3] / 2.0 / m / state[0] /
                                        state[0] +
                                    k * (state[0] - l) * (state[0] - l) / 2.0 -
                                    m * g * state[0] * std::cos(state[1])};
  testing::check_equal_within(system.hamiltonian(state), expected_hamiltonian);
}
}  // namespace

TEST_CASE("unit.systems.ode.elastic_pendulum", "[unit][systems][ode]") {
  run_test_suite();
}