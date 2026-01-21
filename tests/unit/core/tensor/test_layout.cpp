#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/tensor/layout.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
template <class LayoutT>
void test_strides() {
  using layout_t = LayoutT;
  using indexer_t = layout_t::indexer_t;

  constexpr indexer_t order{layout_t::order};
  constexpr core::index_t rank{layout_t::rank};

  testing::check_equal(rank, indexer_t{}.size());

  const auto extents = [&]() {
    constexpr core::index_t min{1}, max{10};
    core::random::RngState rng{testing::random_seed()};
    indexer_t result{};
    for (core::index_t j{0}; j < rank; ++j) {
      result[j] = rng.uniform_int(min, max);
    }
    return result;
  }();

  const auto strides{layout_t::strides_from_extents(extents)};

  const auto expected_strides = [&]() {
    indexer_t result{};
    result[order[0]] = 1;
    for (core::index_t j{0}; j < rank - 1; ++j) {
      result[order[j + 1]] = extents[order[j]] * result[order[j]];
    }
    return result;
  }();

  for (core::index_t j{0}; j < rank; ++j) {
    testing::check_equal(strides[j], expected_strides[j]);
  }
}

void test_strides_implementation() {
  test_strides<core::tensor::Layout<0>>();

  test_strides<core::tensor::Layout<0, 1>>();
  test_strides<core::tensor::Layout<1, 0>>();

  test_strides<core::tensor::Layout<0, 1, 2>>();
  test_strides<core::tensor::Layout<0, 2, 1>>();
  test_strides<core::tensor::Layout<1, 0, 2>>();
  test_strides<core::tensor::Layout<1, 2, 0>>();
  test_strides<core::tensor::Layout<2, 0, 1>>();
  test_strides<core::tensor::Layout<2, 1, 0>>();

  test_strides<core::tensor::Layout<0, 1, 2, 3>>();
  test_strides<core::tensor::Layout<0, 1, 3, 2>>();
  test_strides<core::tensor::Layout<0, 2, 1, 3>>();
  test_strides<core::tensor::Layout<0, 2, 3, 1>>();
  test_strides<core::tensor::Layout<0, 3, 1, 2>>();
  test_strides<core::tensor::Layout<0, 3, 2, 1>>();
  test_strides<core::tensor::Layout<1, 0, 2, 3>>();
  test_strides<core::tensor::Layout<1, 0, 3, 2>>();
  test_strides<core::tensor::Layout<1, 2, 0, 3>>();
  test_strides<core::tensor::Layout<1, 2, 3, 0>>();
  test_strides<core::tensor::Layout<1, 3, 0, 2>>();
  test_strides<core::tensor::Layout<1, 3, 2, 0>>();
  test_strides<core::tensor::Layout<2, 0, 1, 3>>();
  test_strides<core::tensor::Layout<2, 0, 3, 1>>();
  test_strides<core::tensor::Layout<2, 1, 0, 3>>();
  test_strides<core::tensor::Layout<2, 1, 3, 0>>();
  test_strides<core::tensor::Layout<2, 3, 0, 1>>();
  test_strides<core::tensor::Layout<2, 3, 1, 0>>();
  test_strides<core::tensor::Layout<3, 0, 1, 2>>();
  test_strides<core::tensor::Layout<3, 0, 2, 1>>();
  test_strides<core::tensor::Layout<3, 1, 0, 2>>();
  test_strides<core::tensor::Layout<3, 1, 2, 0>>();
  test_strides<core::tensor::Layout<3, 2, 0, 1>>();
  test_strides<core::tensor::Layout<3, 2, 1, 0>>();
}

void test_constexpr_strides() {
  constexpr auto strides{
      core::tensor::Layout<1, 0>::strides_from_extents(3, 4)};
  testing::check_equal(strides[0], core::index_t{4});
  testing::check_equal(strides[1], core::index_t{1});
}

void test_invalid_orders() {
  testing::check_false(core::tensor::LayoutOrder<>);
  testing::check_false(core::tensor::LayoutOrder<0, 2>);
  testing::check_false(core::tensor::LayoutOrder<1, 1, 2>);
  testing::check_false(core::tensor::LayoutOrder<2, 3, 1>);
}

void run_test_suite() {
  test_strides_implementation();
  test_constexpr_strides();
  test_invalid_orders();
}
}  // namespace

TEST_CASE("unit.core.tensor.layout", "[unit][core][tensor]") {
  run_test_suite();
}