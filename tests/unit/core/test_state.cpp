#include "core/state.hpp"
#include "framework/check_equal.hpp"

namespace {
void run_test_suite() {
  using state_t = core::State<4>;

  const state_t state_1{1.0, 2.0, 3.0, 4.0};
  const state_t state_2{2.5, -3.0, 4.0, 1.2};

  testing::check_equal_within(state_1 + state_2, state_t{3.5, -1.0, 7.0, 5.2});
  testing::check_equal_within(3.12 * state_1, state_t{3.12, 6.24, 9.36, 12.48});
  testing::check_equal_within(state_2 * -1.24,
                              state_t{-3.1, 3.72, -4.96, -1.488});
}
}  // namespace

TEST_CASE("unit.core.state", "[unit][core]") { run_test_suite(); }