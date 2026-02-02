#include <cstddef>
#include <type_traits>
#include <utility>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/tensor/buffer.hpp"
#include "pinn/core/tensor/layout.hpp"
#include "pinn/core/tensor/view.hpp"
#include "pinn/core/types.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
template <class LayoutT>
core::index_t expected_offset_from_extents(
    const typename LayoutT::indexer_t& extents,
    const typename LayoutT::indexer_t& index) {
  const auto strides{LayoutT::strides_from_extents(extents)};
  core::index_t result{0};
  for (core::index_t j{0}; j < extents.size(); ++j) {
    result += strides[j] * index[j];
  }
  return result;
}

void test_construction_and_indexing() {
  using index_t = core::index_t;

  constexpr index_t ext_0{2}, ext_1{4}, ext_2{3}, ext_3{5};
  constexpr index_t size{ext_0 * ext_1 * ext_2 * ext_3};

  const double mean{0.0}, stddev{3.0};
  core::random::RngState rng{testing::random_seed()};

  double buffer[size];
  for (index_t j{0}; j < size; ++j) {
    buffer[j] = rng.normal(mean, stddev);
  }

  using layout = core::tensor::Layout<0, 1, 3, 2>;
  constexpr layout::indexer_t extents{ext_0, ext_1, ext_2, ext_3};

  const core::tensor::View<layout, double> view{
      core::tensor::make_view<layout>(buffer, extents)};

  for (index_t j0{0}; j0 < ext_0; ++j0) {
    for (index_t j1{0}; j1 < ext_1; ++j1) {
      for (index_t j2{0}; j2 < ext_2; ++j2) {
        for (index_t j3{0}; j3 < ext_3; ++j3) {
          const auto offset{expected_offset_from_extents<layout>(
              extents, layout::indexer_t{j0, j1, j2, j3})};
          testing::check_equal_within(view(j0, j1, j2, j3), buffer[offset]);

          const double new_value{rng.normal(mean, stddev)};
          view(j0, j1, j2, j3) = new_value;
          testing::check_equal_within(view(j0, j1, j2, j3), new_value);
        }
      }
    }
  }

  using layout_scalar = core::tensor::Layout<>;
  testing::check_equal(layout_scalar::rank, core::index_t{0});

  auto scalar{core::tensor::make_view(buffer)};
  testing::check_equal_within(scalar(), buffer[0]);

  auto another_scalar{core::tensor::make_view<double>(buffer)};
  scalar() = rng.normal(mean, stddev);
  testing::check_equal_within(another_scalar(), scalar());
}

void test_const_correctness() {
  using layout = core::tensor::Layout<0, 1, 2>;

  SECTION("Check access of view constructed from raw pointer.") {
    using writable = core::tensor::View<layout, double>;
    using writable_access = decltype(std::declval<writable>()(0, 0, 0));
    testing::check_true(std::is_same_v<writable_access, double&>);
    testing::check_true(std::is_assignable_v<writable_access, double>);

    // const views *can* mutate underlying data by design.
    testing::check_true(
        std::is_assignable_v<decltype(std::declval<const writable>()(0, 0, 0)),
                             double>);

    using readonly = core::tensor::View<layout, const double>;
    using readonly_access = decltype(std::declval<readonly>()(0, 0, 0));
    testing::check_true(std::is_same_v<readonly_access, const double&>);
    testing::check_false(std::is_assignable_v<readonly_access, double>);
  }

  SECTION("Check access of view constructed from buffer.") {
    constexpr layout::indexer_t extents{2, 3, 4};
    const core::index_t size{extents[0] * extents[1] * extents[2]};

    core::tensor::Buffer<double> buffer{size};
    auto view{core::tensor::make_view<layout>(buffer, extents)};
    using access = decltype(view(0, 0, 0));
    testing::check_true(std::is_same_v<access, double&>);
    testing::check_true(std::is_assignable_v<access, double>);

    const core::tensor::Buffer<double> cbuffer{size};
    auto cview{core::tensor::make_view<layout>(cbuffer, extents)};
    using caccess = decltype(cview(0, 0, 0));
    testing::check_true(std::is_same_v<caccess, const double&>);
    testing::check_false(std::is_assignable_v<caccess, double>);
  }
}

void test_copy_and_move_semantics() {
  using view = core::tensor::View<core::tensor::Layout<0>, double>;
  testing::check_true(std::is_copy_constructible_v<view>);
  testing::check_true(std::is_copy_assignable_v<view>);
  testing::check_true(std::is_move_constructible_v<view>);
  testing::check_true(std::is_move_assignable_v<view>);
  testing::check_true(std::is_trivially_copyable_v<view>);
}

void test_reference_semantics() {
  using view = core::tensor::View<core::tensor::Layout<0, 1>, double>;
  testing::check_true(
      std::is_reference_v<decltype(std::declval<view>()(0, 0))>);
  testing::check_true(
      std::is_lvalue_reference_v<decltype(std::declval<view>()(0, 0))>);
}

void test_size_and_alignment() {
  using layout = core::tensor::Layout<0, 2, 1, 3>;
  using view = core::tensor::View<layout, double>;

  constexpr std::size_t expected_size{sizeof(double*) +
                                      2 * sizeof(layout::indexer_t)};

  testing::check_equal(sizeof(view), expected_size);
  testing::check_true(alignof(view) >= alignof(double*));
}

void test_noexcept_properties() {
  using view = core::tensor::View<core::tensor::Layout<1, 0, 2>, double>;
  testing::check_true(noexcept(std::declval<view>()(0, 0, 0)));
  testing::check_true(std::is_nothrow_copy_constructible_v<view>);
  testing::check_true(std::is_nothrow_copy_assignable_v<view>);
  testing::check_true(std::is_nothrow_move_constructible_v<view>);
  testing::check_true(std::is_nothrow_move_assignable_v<view>);
}

void test_additional_properties() {
  using view = core::tensor::View<core::tensor::Layout<2, 0, 1, 3>, double>;
  testing::check_true(std::is_standard_layout_v<view>);
  testing::check_true(std::is_aggregate_v<view>);
  testing::check_true(std::is_trivially_destructible_v<view>);
}

void run_test_suite() {
  test_construction_and_indexing();
  test_const_correctness();
  test_copy_and_move_semantics();
  test_reference_semantics();
  test_size_and_alignment();
  test_noexcept_properties();
  test_additional_properties();
}
}  // namespace

TEST_CASE("unit.core.tensor.view", "[unit][core][tensor]") { run_test_suite(); }