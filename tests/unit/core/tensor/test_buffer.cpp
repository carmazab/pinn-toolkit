#include <type_traits>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/tensor/buffer.hpp"
#include "pinn/core/types.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
void test_construction_and_assignment() {
  core::random::RngState rng{testing::random_seed()};
  constexpr core::index_t min{10}, max{100};

  constexpr core::index_t zero_size{0};
  using buffer_t = core::tensor::Buffer<double>;

  // Test default constructor.
  buffer_t empty{};
  testing::check_equal(empty.size(), zero_size);
  testing::check_true(empty.data() == nullptr);

  // Test parameter constructor.
  const buffer_t buffer_0{zero_size};
  testing::check_equal(buffer_0.size(), zero_size);
  testing::check_true(buffer_0.data() == nullptr);

  const core::index_t size_1{rng.uniform_int(min, max)};
  buffer_t buffer_1{size_1};
  buffer_t::data_t* const data_1{buffer_1.data()};
  testing::check_equal(buffer_1.size(), size_1);
  testing::check_false(buffer_1.data() == nullptr);

  // Test move constructor.
  buffer_t buffer_2{std::move(buffer_1)};
  testing::check_equal(buffer_1.size(), zero_size);
  testing::check_true(buffer_1.data() == nullptr);
  testing::check_equal(buffer_2.size(), size_1);
  testing::check_equal(buffer_2.data(), data_1);

  // Test move assignment.
  const core::index_t size_3{rng.uniform_int(min, max)};
  buffer_t buffer_3{size_3};
  buffer_3 = std::move(buffer_2);
  testing::check_equal(buffer_2.size(), zero_size);
  testing::check_true(buffer_2.data() == nullptr);
  testing::check_equal(buffer_3.size(), size_1);
  testing::check_equal(buffer_3.data(), data_1);

  // Test swap.
  const core::index_t size_4{rng.uniform_int(min, max)};
  buffer_t buffer_4{size_4};
  buffer_t::data_t* const data_4{buffer_4.data()};
  core::tensor::swap(buffer_3, buffer_4);
  testing::check_equal(buffer_4.size(), size_1);
  testing::check_equal(buffer_4.data(), data_1);
  testing::check_equal(buffer_3.size(), size_4);
  testing::check_equal(buffer_3.data(), data_4);

  // Test self-move.
  buffer_4 = std::move(buffer_4);
  testing::check_equal(buffer_4.size(), size_1);
  testing::check_equal(buffer_4.data(), data_1);

  // Test zero move.
  buffer_4 = std::move(empty);
  testing::check_equal(empty.size(), zero_size);
  testing::check_true(empty.data() == nullptr);
  testing::check_equal(buffer_4.size(), zero_size);
  testing::check_true(buffer_4.data() == nullptr);
}

void test_no_aliasing() {
  using buffer_t = core::tensor::Buffer<double>;
  constexpr core::index_t size{4};

  buffer_t buffer_1{size};
  buffer_t buffer_2{size};
  testing::check_false(buffer_1.data() == buffer_2.data());
}

void test_const_correctness() {
  using buffer_t = core::tensor::Buffer<double>;
  constexpr core::index_t size{4};

  buffer_t buffer{size};
  testing::check_true(std::is_same_v<decltype(buffer.data()), double*>);

  const buffer_t cbuffer{size};
  testing::check_true(std::is_same_v<decltype(cbuffer.data()), const double*>);
}

void test_copy_and_move_semantics() {
  using buffer_t = core::tensor::Buffer<double>;
  testing::check_false(std::is_copy_constructible_v<buffer_t>);
  testing::check_false(std::is_copy_assignable_v<buffer_t>);
  testing::check_true(std::is_move_constructible_v<buffer_t>);
  testing::check_true(std::is_move_assignable_v<buffer_t>);
}

void test_noexcept_properties() {
  using buffer_t = core::tensor::Buffer<double>;
  testing::check_true(std::is_nothrow_move_constructible_v<buffer_t>);
  testing::check_true(std::is_nothrow_move_assignable_v<buffer_t>);
  testing::check_true(
      noexcept(std::declval<buffer_t&>().swap(std::declval<buffer_t&>())));
}

void test_additional_properties() {
  using buffer_t = core::tensor::Buffer<double>;
  testing::check_true(std::is_standard_layout_v<buffer_t>);
  testing::check_false(std::is_trivially_destructible_v<buffer_t>);
}

void run_test_suite() {
  test_construction_and_assignment();
  test_no_aliasing();
  test_const_correctness();
  test_copy_and_move_semantics();
  test_noexcept_properties();
  test_additional_properties();
}
}  // namespace

TEST_CASE("unit.core.tensor.buffer", "[unit][core][tensor]") {
  run_test_suite();
}