#include <cmath>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/systems/ode/elastic_pendulum.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
void run_test_suite() {
  using system_t = systems::ode::ElasticPendulum;
  using state_t = system_t::state_t;
  using data_t = state_t::data_t;

  core::random::RngState rng{testing::random_seed()};

  const data_t min{0.0}, max{3.0};
  const data_t m{rng.uniform(min, max)};
  const data_t g{rng.uniform(min, max)};
  const data_t k{rng.uniform(min, max)};
  const data_t l{rng.uniform(min, max)};
  system_t system{m, g, k, l};

  const data_t mean{0.0}, stddev{3.0};
  const data_t unused_time{rng.normal(mean, stddev)};
  state_t state{rng.normal(mean, stddev), rng.normal(mean, stddev),
                rng.normal(mean, stddev), rng.normal(mean, stddev)};

  const state_t expected_dudt{
      state[2] / m, state[3] / m / state[0] / state[0],
      state[3] * state[3] / m / state[0] / state[0] / state[0] -
          k * (state[0] - l) + m * g * std::cos(state[1]),
      -m * g * state[0] * std::sin(state[1])};
  testing::check_equal_within(system.dudt(unused_time, state), expected_dudt);

  const data_t expected_hamiltonian{state[2] * state[2] / 2.0 / m +
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