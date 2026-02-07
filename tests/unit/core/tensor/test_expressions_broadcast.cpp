#include "pinn/core/tensor/expressions/broadcast.hpp"
#include "pinn/core/tensor/layout.hpp"
#include "pinn/core/tensor/view.hpp"
#include "pinn/core/types.hpp"
#include "testing/check_equal.hpp"
#include "testing/check_exception.hpp"

namespace {
template <class ExpectedLayout, class Lhs, class Rhs>
void test_rank_and_extents(
    const Lhs& lhs, const Rhs& rhs,
    const typename ExpectedLayout::shape_t& expected_extents) {
  const core::index_t expected_rank{ExpectedLayout::rank};

  using result = core::tensor::expressions::broadcast_result<Lhs, Rhs>;
  testing::check_equal(result::broadcast_rank, expected_rank);

  const auto extents{result::broadcast_extents(lhs, rhs)};
  testing::check_equal(extents.size(), expected_rank);

  for (core::index_t j{0}; j < expected_rank; ++j) {
    testing::check_equal(extents, expected_extents);
  }
}

void test_broadcast_result() {
  // Test result of scalar and scalar.
  using scalar_expr = core::tensor::expressions::Scalar<double>;
  const scalar_expr scalar_1{4.0};
  const scalar_expr scalar_2{3.0};

  using layout_0 = core::tensor::Layout<>;
  test_rank_and_extents<layout_0>(scalar_1, scalar_2, layout_0::shape_t{});

  // Test result of view and scalar, scalar and view, and equal shape views.
  using layout_5d = core::tensor::Layout<2, 3, 0, 4, 1>;

  double buffer[120]{};
  const layout_5d::shape_t extents_5d{3, 2, 4, 1, 5};
  const auto view_5d1{core::tensor::make_view<layout_5d>(buffer, extents_5d)};
  const auto view_5d2{core::tensor::make_view<layout_5d>(buffer, extents_5d)};
  const auto view_5d3{core::tensor::make_view<layout_5d>(
      buffer, layout_5d::shape_t{1, 2, 1, 4, 5})};

  test_rank_and_extents<layout_5d>(view_5d1, scalar_1, extents_5d);
  test_rank_and_extents<layout_5d>(scalar_2, view_5d2, extents_5d);
  test_rank_and_extents<layout_5d>(view_5d1, view_5d2, extents_5d);
  test_rank_and_extents<layout_5d>(view_5d2, view_5d3,
                                   layout_5d::shape_t{3, 2, 4, 4, 5});

  // Test result of views of different shapes.
  using layout_3d = core::tensor::Layout<0, 1, 2>;

  const layout_3d::shape_t extents_3d{1, 5, 5};
  const auto view_3d{core::tensor::make_view<layout_3d>(buffer, extents_3d)};

  const layout_5d::shape_t expected_extents{3, 2, 4, 5, 5};
  test_rank_and_extents<layout_5d>(view_5d1, view_3d, expected_extents);
  test_rank_and_extents<layout_5d>(view_3d, view_5d2, expected_extents);

  // Test result involving 1-dimensional views.
  using layout_1d = core::tensor::Layout<0>;

  const auto view_1d1{
      core::tensor::make_view<layout_1d>(buffer, layout_1d::shape_t{1})};
  const auto view_1d2{
      core::tensor::make_view<layout_1d>(buffer, layout_1d::shape_t{5})};

  test_rank_and_extents<layout_1d>(view_1d1, view_1d2, layout_1d::shape_t{5});
  test_rank_and_extents<layout_1d>(view_1d2, view_1d1, layout_1d::shape_t{5});
  test_rank_and_extents<layout_5d>(view_1d1, view_5d1, extents_5d);
  test_rank_and_extents<layout_3d>(view_3d, view_1d2, extents_3d);

  // Test singleton.
  const auto singleton{core::tensor::make_view<layout_5d>(
      buffer, layout_5d::shape_t{1, 1, 1, 1, 1})};

  test_rank_and_extents<layout_5d>(singleton, view_5d1, extents_5d);
  test_rank_and_extents<layout_5d>(view_5d1, singleton, extents_5d);
}

void test_invalid_shapes() {
  using layout = core::tensor::Layout<0, 1, 2, 3>;
  using view = core::tensor::View<double, layout>;

  double buffer[24]{};
  const auto view_1{
      core::tensor::make_view<layout>(buffer, layout::shape_t{2, 3, 1, 4})};
  const auto view_2{
      core::tensor::make_view<layout>(buffer, layout::shape_t{1, 2, 3, 4})};

  using result = core::tensor::expressions::broadcast_result<view, view>;
  testing::check_throws_as_logic_error(
      [&]() { result::broadcast_extents(view_1, view_2); });
}

void run_test_suite() {
  test_broadcast_result();
  test_invalid_shapes();
}
}  // namespace

TEST_CASE("unit.core.tensor.expressions.broadcast",
          "[unit][core][tensor][expressions]") {
  run_test_suite();
}