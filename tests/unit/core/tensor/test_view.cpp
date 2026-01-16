#include <array>
#include <type_traits>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/tensor/buffer.hpp"
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

void test_slicing() {
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
  using writable = core::tensor::View<double, 3>;
  using writable_access = decltype(std::declval<writable>()(0, 0, 0));
  testing::check_true(std::is_same_v<writable_access, double&>);
  testing::check_true(std::is_assignable_v<writable_access, double>);

  // const views *can* mutate underlying data by design.
  testing::check_true(
      std::is_assignable_v<decltype(std::declval<const writable>()(0, 0, 0)),
                           double>);

  using readonly_access =
      decltype(std::declval<core::tensor::View<const double, 3>>()(0, 0, 0));
  testing::check_true(std::is_same_v<readonly_access, const double&>);
  testing::check_false(std::is_assignable_v<readonly_access, double>);

  // Check access of view constructed from buffer.
  const std::array<core::index_t, 3> extents{2, 3, 4};

  core::tensor::Buffer<double> buffer{6};
  auto view{core::tensor::make_view(buffer, extents)};
  using access = decltype(view(0, 0, 0));
  testing::check_true(std::is_same_v<access, double&>);
  testing::check_true(std::is_assignable_v<access, double>);

  const core::tensor::Buffer<double> cbuffer{6};
  auto cview{core::tensor::make_view(cbuffer, extents)};
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