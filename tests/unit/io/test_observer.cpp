#include "pinn/core/types.hpp"
#include "pinn/io/observer.hpp"
#include "testing/check_equal.hpp"
#include "testing/helpers.hpp"

namespace {
void run_test_suite() {
  const core::index_t write_every{3};
  testing::helpers::DummyWriter writer;
  io::Observer observer{writer, write_every};

  using state_t = testing::helpers::DummyWriter::state_t;

  double time{0.123};
  state_t state{0.1, 0.2, 0.3, 0.4};
  observer(time, state, 0);

  testing::check_equal_within(writer.t.back(), time);
  for (core::index_t s{0}; s < state_t::dof; ++s) {
    testing::check_equal_within(writer.history[s].back(), state[s]);
  }

  const core::index_t iters{24};
  for (core::index_t n{0}; n < iters; ++n) {
    state = 2.0 * state;
    time += 0.1;
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