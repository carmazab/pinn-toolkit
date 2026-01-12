#include <array>
#include <type_traits>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/tensor/view.hpp"
#include "pinn/core/types.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
template <class ArrayT>
ArrayT strides_from_extents(const ArrayT& extents) {
  ArrayT result;
  result[0] = 1;
  for (core::index_t j{0}; j < extents.size() - 1; ++j) {
    result[j + 1] = extents[j] * result[j];
  }
  return result;
}

template <class ArrayT>
core::index_t offset_from_extents(const ArrayT& extents, const ArrayT& index) {
  const auto strides{strides_from_extents(extents)};
  core::index_t result{0};
  for (core::index_t j{0}; j < extents.size(); ++j) {
    result += strides[j] * index[j];
  }
  return result;
}

void test_construction_and_data_manipulation() {
  using index_t = core::index_t;
  constexpr index_t rank{4};
  constexpr index_t ext_1{3}, ext_2{5}, ext_3{4}, ext_4{6};
  constexpr index_t size{ext_1 * ext_2 * ext_3 * ext_4};
  using indexer_t = std::array<index_t, rank>;

  const double mean{0.0}, stddev{3.0};
  core::random::RngState rng{testing::random_seed()};

  double buffer[size];
  for (index_t j{0}; j < size; ++j) {
    buffer[j] = rng.normal(mean, stddev);
  }

  const indexer_t extents{ext_1, ext_2, ext_3, ext_4};
  const auto view{core::tensor::make_view(buffer, extents)};

  for (index_t j1{0}; j1 < ext_1; ++j1) {
    for (index_t j2{0}; j2 < ext_2; ++j2) {
      for (index_t j3{0}; j3 < ext_3; ++j3) {
        for (index_t j4{0}; j4 < ext_4; ++j4) {
          testing::check_equal_within(
              view(j1, j2, j3, j4),
              buffer[offset_from_extents(extents, indexer_t{j1, j2, j3, j4})]);
        }
      }
    }
  }

  const double random_value{rng.normal(mean, stddev)};
  view(1, 2, 1, 4) = random_value;
  testing::check_equal_within(view(1, 2, 1, 4), random_value);
}

void test_const_correctness() {
  using writable = core::tensor::View<double, 3>;
  using readonly = core::tensor::View<const double, 3>;

  testing::check_true(
      std::is_same_v<decltype(std::declval<writable>()(0, 0, 0)), double&>);
  testing::check_true(
      std::is_same_v<decltype(std::declval<readonly>()(0, 0, 0)),
                     const double&>);

  testing::check_true(
      std::is_assignable_v<decltype(std::declval<writable>()(0, 0, 0)),
                           double>);
  testing::check_false(
      std::is_assignable_v<decltype(std::declval<readonly>()(0, 0, 0)),
                           double>);

  // const views *can* mutate underlying data by design.
  testing::check_true(
      std::is_assignable_v<decltype(std::declval<const writable>()(0, 0, 0)),
                           double>);
}

void test_copy_and_move_semantics() {
  using view = core::tensor::View<double, 1>;
  testing::check_true(std::is_copy_constructible_v<view>);
  testing::check_true(std::is_copy_assignable_v<view>);
  testing::check_true(std::is_move_constructible_v<view>);
  testing::check_true(std::is_move_assignable_v<view>);
  testing::check_true(std::is_trivially_copyable_v<view>);
}

void test_reference_semantics() {
  using view = core::tensor::View<double, 2>;
  testing::check_true(
      std::is_reference_v<decltype(std::declval<view>()(0, 0))>);
  testing::check_true(
      std::is_lvalue_reference_v<decltype(std::declval<view>()(0, 0))>);
}

void test_size_and_alignment() {
  using view = core::tensor::View<double, 4>;

  constexpr std::size_t expected_size =
      sizeof(double*) + 2 * sizeof(std::array<core::index_t, 4>);

  testing::check_equal(sizeof(view), expected_size);
  testing::check_true(alignof(view) >= alignof(double*));
}

void test_noexcept_properties() {
  using view = core::tensor::View<double, 3>;
  testing::check_true(noexcept(std::declval<view>()(0, 0, 0)));
  testing::check_true(std::is_nothrow_copy_constructible_v<view>);
  testing::check_true(std::is_nothrow_copy_assignable_v<view>);
  testing::check_true(std::is_nothrow_move_constructible_v<view>);
  testing::check_true(std::is_nothrow_move_assignable_v<view>);
}

void test_additional_properties() {
  using view = core::tensor::View<double, 4>;
  testing::check_true(std::is_standard_layout_v<view>);
  testing::check_true(std::is_aggregate_v<view>);
  testing::check_true(std::is_trivially_destructible_v<view>);
}

void run_test_suite() {
  test_construction_and_data_manipulation();
  test_const_correctness();
  test_copy_and_move_semantics();
  test_reference_semantics();
  test_size_and_alignment();
  test_noexcept_properties();
  test_additional_properties();
}
}  // namespace

TEST_CASE("unit.core.tensor.view", "[unit][core][tensor]") { run_test_suite(); }