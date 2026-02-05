#include <type_traits>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/tensor/buffer.hpp"
#include "pinn/core/tensor/layout.hpp"
#include "pinn/core/tensor/view.hpp"
#include "pinn/core/types.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
void test_permute() {
  using index_t = core::index_t;

  constexpr index_t ext_0{2}, ext_1{4}, ext_2{3}, ext_3{1};
  constexpr index_t size{ext_0 * ext_1 * ext_2 * ext_3};

  const double mean{0.0}, stddev{3.0};
  core::random::RngState rng{testing::random_seed()};

  double buffer[size];
  for (index_t j{0}; j < size; ++j) {
    buffer[j] = rng.normal(mean, stddev);
  }

  using layout = core::tensor::Layout<1, 3, 0, 2>;
  constexpr index_t rank{layout::rank};
  constexpr layout::shape_t extents{ext_0, ext_1, ext_2, ext_3};

  const core::tensor::View<double, layout> view{
      core::tensor::make_view<layout>(buffer, extents)};

  {  // Check that identity permutation does not change layout.
    using expected_layout = layout;
    const core::tensor::View<double, expected_layout> permuted{
        view.permute<0, 1, 2, 3>()};

    layout::shape_t map{0, 1, 2, 3};
    for (index_t j{0}; j < rank; ++j) {
      testing::check_equal(permuted.extents[j], view.extents[map[j]]);
      testing::check_equal(permuted.strides[j], view.strides[map[j]]);
    }

    for (index_t j0{0}; j0 < ext_0; ++j0) {
      for (index_t j1{0}; j1 < ext_1; ++j1) {
        for (index_t j2{0}; j2 < ext_2; ++j2) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(view(j0, j1, j2, j3),
                                        permuted(j0, j1, j2, j3));

            const double new_value{rng.normal(mean, stddev)};
            permuted(j0, j1, j2, j3) = new_value;
            testing::check_equal_within(view(j0, j1, j2, j3), new_value);
          }
        }
      }
    }
  }

  {  // Check reversed layout.
    using expected_layout = core::tensor::Layout<2, 0, 3, 1>;
    const core::tensor::View<double, expected_layout> reversed{
        view.reverse_layout()};

    layout::shape_t map{3, 2, 1, 0};
    for (index_t j{0}; j < rank; ++j) {
      testing::check_equal(reversed.extents[j], view.extents[map[j]]);
      testing::check_equal(reversed.strides[j], view.strides[map[j]]);
    }

    for (index_t j0{0}; j0 < ext_0; ++j0) {
      for (index_t j1{0}; j1 < ext_1; ++j1) {
        for (index_t j2{0}; j2 < ext_2; ++j2) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(view(j0, j1, j2, j3),
                                        reversed(j3, j2, j1, j0));

            const double new_value{rng.normal(mean, stddev)};
            reversed(j3, j2, j1, j0) = new_value;
            testing::check_equal_within(view(j0, j1, j2, j3), new_value);
          }
        }
      }
    }
  }

  {  // Check transposition of two axes.
    using expected_layout = core::tensor::Layout<1, 2, 0, 3>;
    const core::tensor::View<double, expected_layout> transposed{
        view.transpose<1, 3>()};

    layout::shape_t map{0, 3, 2, 1};
    for (index_t j{0}; j < rank; ++j) {
      testing::check_equal(transposed.extents[j], view.extents[map[j]]);
      testing::check_equal(transposed.strides[j], view.strides[map[j]]);
    }

    for (index_t j0{0}; j0 < ext_0; ++j0) {
      for (index_t j1{0}; j1 < ext_1; ++j1) {
        for (index_t j2{0}; j2 < ext_2; ++j2) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(view(j0, j1, j2, j3),
                                        transposed(j0, j3, j2, j1));

            const double new_value{rng.normal(mean, stddev)};
            transposed(j0, j3, j2, j1) = new_value;
            testing::check_equal_within(view(j0, j1, j2, j3), new_value);
          }
        }
      }
    }
  }

  {  // Check nontrivial permutation.
    using expected_layout = core::tensor::Layout<0, 2, 3, 1>;
    const core::tensor::View<double, expected_layout> permuted{
        view.permute<2, 3, 1, 0>()};

    layout::shape_t map{2, 3, 1, 0};
    for (index_t j{0}; j < rank; ++j) {
      testing::check_equal(permuted.extents[j], view.extents[map[j]]);
      testing::check_equal(permuted.strides[j], view.strides[map[j]]);
    }

    for (index_t j0{0}; j0 < ext_0; ++j0) {
      for (index_t j1{0}; j1 < ext_1; ++j1) {
        for (index_t j2{0}; j2 < ext_2; ++j2) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(view(j0, j1, j2, j3),
                                        permuted(j2, j3, j1, j0));

            const double new_value{rng.normal(mean, stddev)};
            permuted(j2, j3, j1, j0) = new_value;
            testing::check_equal_within(view(j0, j1, j2, j3), new_value);
          }
        }
      }
    }
  }

  {  // Check another nontrivial permutation.
    using expected_layout = core::tensor::Layout<1, 0, 2, 3>;
    const core::tensor::View<double, expected_layout> permuted{
        view.permute<0, 2, 3, 1>()};

    layout::shape_t map{0, 2, 3, 1};
    for (index_t j{0}; j < rank; ++j) {
      testing::check_equal(permuted.extents[j], view.extents[map[j]]);
      testing::check_equal(permuted.strides[j], view.strides[map[j]]);
    }

    for (index_t j0{0}; j0 < ext_0; ++j0) {
      for (index_t j1{0}; j1 < ext_1; ++j1) {
        for (index_t j2{0}; j2 < ext_2; ++j2) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(view(j0, j1, j2, j3),
                                        permuted(j0, j2, j3, j1));

            const double new_value{rng.normal(mean, stddev)};
            permuted(j0, j2, j3, j1) = new_value;
            testing::check_equal_within(view(j0, j1, j2, j3), new_value);
          }
        }
      }
    }
  }
}

void test_slice() {
  using index_t = core::index_t;

  constexpr index_t ext_0{2}, ext_1{4}, ext_2{3}, ext_3{1};
  constexpr index_t size{ext_0 * ext_1 * ext_2 * ext_3};

  const double mean{0.0}, stddev{3.0};
  core::random::RngState rng{testing::random_seed()};

  double buffer[size];
  for (index_t j{0}; j < size; ++j) {
    buffer[j] = rng.normal(mean, stddev);
  }

  using layout = core::tensor::Layout<1, 3, 0, 2>;
  constexpr index_t rank{layout::rank};
  constexpr layout::shape_t extents{ext_0, ext_1, ext_2, ext_3};

  const core::tensor::View<double, layout> view{
      core::tensor::make_view<layout>(buffer, extents)};

  {  // Check slicing along first axis.
    using expected_layout = core::tensor::Layout<2, 0, 1>;

    for (index_t slice_index{0}; slice_index < extents[0]; ++slice_index) {
      const core::tensor::View<double, expected_layout> subview{
          view.slice<0>(slice_index)};

      for (index_t j{0}; j < rank - 1; ++j) {
        index_t k{j < 0 ? j : j + 1};
        testing::check_equal(subview.extents[j], view.extents[k]);
        testing::check_equal(subview.strides[j], view.strides[k]);
      }

      for (index_t j1{0}; j1 < ext_1; ++j1) {
        for (index_t j2{0}; j2 < ext_2; ++j2) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(subview(j1, j2, j3),
                                        view(slice_index, j1, j2, j3));

            const double new_value{rng.normal(mean, stddev)};
            subview(j1, j2, j3) = new_value;
            testing::check_equal_within(view(slice_index, j1, j2, j3),
                                        new_value);
          }
        }
      }
    }
  }

  {  // Check slicing along second axis.
    using expected_layout = core::tensor::Layout<1, 0, 2>;

    for (index_t slice_index{0}; slice_index < extents[1]; ++slice_index) {
      const core::tensor::View<double, expected_layout> subview{
          view.slice<1>(slice_index)};

      for (index_t j{0}; j < rank - 1; ++j) {
        index_t k{j < 1 ? j : j + 1};
        testing::check_equal(subview.extents[j], view.extents[k]);
        testing::check_equal(subview.strides[j], view.strides[k]);
      }

      for (index_t j0{0}; j0 < ext_0; ++j0) {
        for (index_t j2{0}; j2 < ext_2; ++j2) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(subview(j0, j2, j3),
                                        view(j0, slice_index, j2, j3));

            const double new_value{rng.normal(mean, stddev)};
            subview(j0, j2, j3) = new_value;
            testing::check_equal_within(view(j0, slice_index, j2, j3),
                                        new_value);
          }
        }
      }
    }
  }

  {  // Check slicing along third axis.
    using expected_layout = core::tensor::Layout<0, 2, 1>;

    for (index_t slice_index{0}; slice_index < extents[2]; ++slice_index) {
      const core::tensor::View<double, expected_layout> subview{
          view.slice<2>(slice_index)};

      for (index_t j{0}; j < rank - 1; ++j) {
        index_t k{j < 2 ? j : j + 1};
        testing::check_equal(subview.extents[j], view.extents[k]);
        testing::check_equal(subview.strides[j], view.strides[k]);
      }

      for (index_t j0{0}; j0 < ext_0; ++j0) {
        for (index_t j1{0}; j1 < ext_1; ++j1) {
          for (index_t j3{0}; j3 < ext_3; ++j3) {
            testing::check_equal_within(subview(j0, j1, j3),
                                        view(j0, j1, slice_index, j3));

            const double new_value{rng.normal(mean, stddev)};
            subview(j0, j1, j3) = new_value;
            testing::check_equal_within(view(j0, j1, slice_index, j3),
                                        new_value);
          }
        }
      }
    }
  }

  {  // Check slicing along fourth axis.
    using expected_layout = core::tensor::Layout<1, 2, 0>;

    for (index_t slice_index{0}; slice_index < extents[3]; ++slice_index) {
      const core::tensor::View<double, expected_layout> subview{
          view.slice<3>(slice_index)};

      for (index_t j{0}; j < rank - 1; ++j) {
        index_t k{j < 3 ? j : j + 1};
        testing::check_equal(subview.extents[j], view.extents[k]);
        testing::check_equal(subview.strides[j], view.strides[k]);
      }

      for (index_t j0{0}; j0 < ext_0; ++j0) {
        for (index_t j1{0}; j1 < ext_1; ++j1) {
          for (index_t j2{0}; j2 < ext_2; ++j2) {
            testing::check_equal_within(subview(j0, j1, j2),
                                        view(j0, j1, j2, slice_index));

            const double new_value{rng.normal(mean, stddev)};
            subview(j0, j1, j2) = new_value;
            testing::check_equal_within(view(j0, j1, j2, slice_index),
                                        new_value);
          }
        }
      }
    }
  }

  {  // Check that slicing a rank-1 view gives a scalar.
    constexpr core::index_t extent{5};
    double another_buffer[extent];
    for (index_t j{0}; j < extent; ++j) {
      another_buffer[j] = rng.normal(mean, stddev);
    }

    using layout_1d = core::tensor::Layout<0>;
    const auto view_1d{core::tensor::make_view<layout_1d>(
        another_buffer, layout_1d::shape_t{extent})};

    using expected_layout = core::tensor::Layout<>;
    const core::index_t index{rng.uniform_int<core::index_t>(0, extent - 1)};
    const core::tensor::View<double, expected_layout> slice{
        view_1d.slice<0>(index)};
    testing::check_equal_within(slice(), another_buffer[index]);
  }
}

void test_compositions() {
  using layout = core::tensor::Layout<3, 4, 1, 0, 2>;
  core::index_t size{4};
  core::tensor::Buffer<double> buffer{size};
  constexpr layout::shape_t extents{1, 2, 1, 2, 1};
  auto view{core::tensor::make_view<layout>(buffer, extents)};

  auto comp_1{view.slice<2>(0).transpose<3, 2>().reverse_layout()};
  using expected_1 = core::tensor::Layout<0, 1, 3, 2>;
  testing::check_true(std::is_same_v<expected_1, decltype(comp_1)::layout_t>);

  auto comp_2{view.permute<2, 0, 1, 4, 3>().transpose<3, 2>().slice<1>(0)};
  using expected_2 = core::tensor::Layout<1, 2, 3, 0>;
  testing::check_true(std::is_same_v<expected_2, decltype(comp_2)::layout_t>);

  auto comp_3{view.slice<0>(0).slice<0>(0).slice<0>(0).reverse_layout()};
  using expected_3 = core::tensor::Layout<1, 0>;
  testing::check_true(std::is_same_v<expected_3, decltype(comp_3)::layout_t>);
}

void test_const_correctness_preservation() {
  using layout = core::tensor::Layout<0, 2, 1>;

  constexpr layout::shape_t extents{2, 3, 4};
  const core::index_t size{extents[0] * extents[1] * extents[2]};

  {  // Test correctness with mutable views.
    core::tensor::Buffer<double> buffer{size};
    auto view{core::tensor::make_view<layout>(buffer, extents)};

    auto reordered{view.permute<1, 2, 0>()};
    using reaccess = decltype(reordered(0, 0, 0));
    testing::check_true(std::is_same_v<reaccess, double&>);
    testing::check_true(std::is_assignable_v<reaccess, double>);

    auto reversed{view.reverse_layout()};
    using revaccess = decltype(reversed(0, 0, 0));
    testing::check_true(std::is_same_v<revaccess, double&>);
    testing::check_true(std::is_assignable_v<revaccess, double>);

    auto subview{view.slice<0>(0)};
    using subaccess = decltype(subview(0, 0));
    testing::check_true(std::is_same_v<subaccess, double&>);
    testing::check_true(std::is_assignable_v<subaccess, double>);

    auto transposed{view.transpose<0, 2>()};
    using traccess = decltype(transposed(0, 0, 0));
    testing::check_true(std::is_same_v<traccess, double&>);
    testing::check_true(std::is_assignable_v<traccess, double>);
  }

  {  // Test correctness with const views.
    const core::tensor::Buffer<double> cbuffer{size};
    auto cview{core::tensor::make_view<layout>(cbuffer, extents)};

    auto creordered{cview.permute<1, 2, 0>()};
    using creaccess = decltype(creordered(0, 0, 0));
    testing::check_true(std::is_same_v<creaccess, const double&>);
    testing::check_false(std::is_assignable_v<creaccess, double>);

    auto creversed{cview.reverse_layout()};
    using crevaccess = decltype(creversed(0, 0, 0));
    testing::check_true(std::is_same_v<crevaccess, const double&>);
    testing::check_false(std::is_assignable_v<crevaccess, double>);

    auto csubview{cview.slice<0>(0)};
    using csubaccess = decltype(csubview(0, 0));
    testing::check_true(std::is_same_v<csubaccess, const double&>);
    testing::check_false(std::is_assignable_v<csubaccess, double>);

    auto ctransposed{cview.transpose<0, 2>()};
    using ctraccess = decltype(ctransposed(0, 0, 0));
    testing::check_true(std::is_same_v<ctraccess, const double&>);
    testing::check_false(std::is_assignable_v<ctraccess, double>);
  }
}

void run_test_suite() {
  test_permute();
  test_slice();
  test_compositions();
  test_const_correctness_preservation();
}
}  // namespace

TEST_CASE("unit.core.tensor.layout_traits", "[unit][core][tensor]") {
  run_test_suite();
}