#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/types.hpp"
#include "pinn/io/observer.hpp"
#include "testing/check_equal.hpp"
#include "testing/helpers.hpp"
#include "testing/random_seed.hpp"

namespace {
void run_test_suite() {
  core::random::RngState rng{testing::random_seed()};

  const core::index_t write_every{rng.uniform_int<core::index_t>(1, 5)};
  testing::helpers::DummyWriter writer;
  io::Observer observer{writer, write_every};

  using state_t = testing::helpers::DummyWriter::state_t;
  using data_t = state_t::data_t;

  const data_t mean{0.0}, stddev{3.0};
  data_t time{rng.normal(mean, stddev)};
  state_t state{rng.normal(mean, stddev), rng.normal(mean, stddev),
                rng.normal(mean, stddev), rng.normal(mean, stddev)};
  observer(time, state, 0);

  testing::check_equal_within(writer.t.back(), time);
  for (core::index_t s{0}; s < state_t::dof; ++s) {
    testing::check_equal_within(writer.history[s].back(), state[s]);
  }

  const core::index_t iters{rng.uniform_int<core::index_t>(10, 30)};
  for (core::index_t n{1}; n < iters + 1; ++n) {
    state = rng.normal(mean, stddev) * state;
    time += rng.normal(mean, stddev);
    observer(time, state, n);

    if (n % write_every == 0) {
      testing::check_equal_within(writer.t.back(), time);
      for (core::index_t s{0}; s < state_t::dof; ++s) {
        testing::check_equal_within(writer.history[s].back(), state[s]);
      }
    }
  }

  core::index_t expected_size{iters / write_every + 1};
  testing::check_equal(writer.t.size(), expected_size);
  for (core::index_t s{0}; s < state_t::dof; ++s) {
    testing::check_equal(writer.history[s].size(), expected_size);
  }
}
}  // namespace

TEST_CASE("unit.io.observer", "[unit][io]") { run_test_suite(); }