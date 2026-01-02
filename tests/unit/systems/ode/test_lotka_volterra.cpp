#include <cmath>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/systems/ode/lotka_volterra.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
void run_test_suite() {
  using system_t = systems::ode::LotkaVolterra;
  using state_t = system_t::state_t;
  using data_t = state_t::data_t;

  core::random::RngState rng{testing::random_seed()};

  const data_t min{0.0}, max{3.0};
  const data_t a{rng.uniform(min, max)};
  const data_t b{rng.uniform(min, max)};
  const data_t c{rng.uniform(min, max)};
  const data_t d{rng.uniform(min, max)};
  system_t system{a, b, c, d};

  const data_t unused_time{rng.uniform(min, max)};
  state_t state{rng.uniform(min, max), rng.uniform(min, max)};

  const state_t expected_dudt{a * state[0] - b * state[0] * state[1],
                              -c * state[1] + d * state[0] * state[1]};
  testing::check_equal_within(system.dudt(unused_time, state), expected_dudt);

  const data_t expected_hamiltonian{d * state[0] - c * std::log(state[0]) +
                                    b * state[1] - a * std::log(state[1])};
  testing::check_equal_within(system.hamiltonian(state), expected_hamiltonian);
}
}  // namespace

TEST_CASE("unit.systems.ode.lotka_volterra", "[unit][systems][ode]") {
  run_test_suite();
}