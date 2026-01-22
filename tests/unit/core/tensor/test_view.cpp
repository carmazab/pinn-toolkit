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
  constexpr index_t ext_1{3}, ext_2{5}, ext_3{4}, ext_4{6};
  constexpr index_t size{ext_1 * ext_2 * ext_3 * ext_4};

  const double mean{0.0}, stddev{3.0};
  core::random::RngState rng{testing::random_seed()};

  double buffer[size];
  for (index_t j{0}; j < size; ++j) {
    buffer[j] = rng.normal(mean, stddev);
  }

  using layout = core::tensor::Layout<0, 1, 3, 2>;
  constexpr layout::indexer_t extents{ext_1, ext_2, ext_3, ext_4};
  const auto view{core::tensor::make_view<layout>(buffer, extents)};

  for (index_t j1{0}; j1 < ext_1; ++j1) {
    for (index_t j2{0}; j2 < ext_2; ++j2) {
      for (index_t j3{0}; j3 < ext_3; ++j3) {
        for (index_t j4{0}; j4 < ext_4; ++j4) {
          const auto offset{expected_offset_from_extents<layout>(
              extents, layout::indexer_t{j1, j2, j3, j4})};
          testing::check_equal_within(view(j1, j2, j3, j4), buffer[offset]);
        }
      }
    }
  }

  const double random_value{rng.normal(mean, stddev)};
  view(1, 2, 1, 4) = random_value;
  testing::check_equal_within(view(1, 2, 1, 4), random_value);
}

void test_slicing() {
  using index_t = core::index_t;
  constexpr index_t ext_1{3}, ext_2{5}, ext_3{4}, ext_4{6};
  constexpr index_t size{ext_1 * ext_2 * ext_3 * ext_4};

  const double mean{0.0}, stddev{3.0};
  core::random::RngState rng{testing::random_seed()};

  double buffer[size];
  for (index_t j{0}; j < size; ++j) {
    buffer[j] = rng.normal(mean, stddev);
  }

  using layout = core::tensor::Layout<1, 2, 0, 3>;
  constexpr core::index_t rank{layout::rank};
  constexpr layout::indexer_t extents{ext_1, ext_2, ext_3, ext_4};
  const auto view{core::tensor::make_view<layout>(buffer, extents)};

  for (index_t slice_index{0}; slice_index < extents[0]; ++slice_index) {
    const auto subview{view.slice<0>(slice_index)};

    for (index_t j{0}; j < rank - 1; ++j) {
      index_t k{j + (j >= 0)};
      testing::check_equal(subview.extents[j], view.extents[k]);
      testing::check_equal(subview.strides[j], view.strides[k]);
    }

    for (index_t j2{0}; j2 < ext_2; ++j2) {
      for (index_t j3{0}; j3 < ext_3; ++j3) {
        for (index_t j4{0}; j4 < ext_4; ++j4) {
          testing::check_equal_within(subview(j2, j3, j4),
                                      view(slice_index, j2, j3, j4));
        }
      }
    }
  }

  for (index_t slice_index{0}; slice_index < extents[1]; ++slice_index) {
    const auto subview{view.slice<1>(slice_index)};

    for (index_t j{0}; j < rank - 1; ++j) {
      index_t k{j + (j >= 1)};
      testing::check_equal(subview.extents[j], view.extents[k]);
      testing::check_equal(subview.strides[j], view.strides[k]);
    }

    for (index_t j1{0}; j1 < ext_1; ++j1) {
      for (index_t j3{0}; j3 < ext_3; ++j3) {
        for (index_t j4{0}; j4 < ext_4; ++j4) {
          testing::check_equal_within(subview(j1, j3, j4),
                                      view(j1, slice_index, j3, j4));
        }
      }
    }
  }

  for (index_t slice_index{0}; slice_index < extents[2]; ++slice_index) {
    const auto subview{view.slice<2>(slice_index)};

    for (index_t j{0}; j < rank - 1; ++j) {
      index_t k{j + (j >= 2)};
      testing::check_equal(subview.extents[j], view.extents[k]);
      testing::check_equal(subview.strides[j], view.strides[k]);
    }

    for (index_t j1{0}; j1 < ext_1; ++j1) {
      for (index_t j2{0}; j2 < ext_2; ++j2) {
        for (index_t j4{0}; j4 < ext_4; ++j4) {
          testing::check_equal_within(subview(j1, j2, j4),
                                      view(j1, j2, slice_index, j4));
        }
      }
    }
  }

  for (index_t slice_index{0}; slice_index < extents[3]; ++slice_index) {
    const auto subview{view.slice<3>(slice_index)};

    for (index_t j{0}; j < rank - 1; ++j) {
      index_t k{j + (j >= 3)};
      testing::check_equal(subview.extents[j], view.extents[k]);
      testing::check_equal(subview.strides[j], view.strides[k]);
    }

    for (index_t j1{0}; j1 < ext_1; ++j1) {
      for (index_t j2{0}; j2 < ext_2; ++j2) {
        for (index_t j3{0}; j3 < ext_3; ++j3) {
          testing::check_equal_within(subview(j1, j2, j3),
                                      view(j1, j2, j3, slice_index));
        }
      }
    }
  }

  // Test no aliasing.
  auto subview{view.slice<1>(2)};
  const double random_value{rng.normal(mean, stddev)};
  subview(1, 1, 1) = random_value;
  testing::check_equal_within(view(1, 2, 1, 1), random_value);
}

void test_const_correctness() {
  using layout = core::tensor::Layout<0, 1, 2>;

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

  // Check access of view constructed from buffer.
  constexpr layout::indexer_t extents{2, 3, 4};

  core::tensor::Buffer<double> buffer{6};
  auto view{core::tensor::make_view<layout>(buffer, extents)};
  using access = decltype(view(0, 0, 0));
  testing::check_true(std::is_same_v<access, double&>);
  testing::check_true(std::is_assignable_v<access, double>);

  const core::tensor::Buffer<double> cbuffer{6};
  auto cview{core::tensor::make_view<layout>(cbuffer, extents)};
  using caccess = decltype(cview(0, 0, 0));
  testing::check_true(std::is_same_v<caccess, const double&>);
  testing::check_false(std::is_assignable_v<caccess, double>);

  // Check that subviews preserve const-correctness.
  auto subview{view.slice<0>(0)};
  using subaccess = decltype(subview(0, 0));
  testing::check_true(std::is_same_v<subaccess, double&>);
  testing::check_true(std::is_assignable_v<subaccess, double>);

  auto csubview{cview.slice<0>(0)};
  using csubaccess = decltype(csubview(0, 0));
  testing::check_true(std::is_same_v<csubaccess, const double&>);
  testing::check_false(std::is_assignable_v<csubaccess, double>);
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
  test_slicing();
  test_const_correctness();
  test_copy_and_move_semantics();
  test_reference_semantics();
  test_size_and_alignment();
  test_noexcept_properties();
  test_additional_properties();
}
}  // namespace

TEST_CASE("unit.core.tensor.view", "[unit][core][tensor]") { run_test_suite(); }