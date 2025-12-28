#pragma once

#include "core/concepts.hpp"
#include "core/types.hpp"
#include "framework/framework.hpp"

namespace testing {
template <class T>
void check_true(T a) {
  CHECK(a);
}

template <class T>
void check_false(T a) {
  CHECK_FALSE(a);
}

template <class T>
void check_equal(T a, T b) {
  check_true(a == b);
}

template <core::concepts::Real R>
void check_equal_within(R a, R b, R rel_tol = static_cast<R>(1.0e-12),
                        R abs_tol = static_cast<R>(1.0e-14)) {
  CHECK_THAT(a, Catch::Matchers::WithinRel(b, rel_tol) ||
                    Catch::Matchers::WithinAbs(b, abs_tol));
}

template <core::concepts::State S>
void check_equal_within(
    const S& a, const S& b,
    typename S::data_t tol = static_cast<typename S::data_t>(1.0e-12)) {
  for (core::index_t j{0}; j < S::dof; ++j) {
    check_equal_within(a[j], b[j]);
  }
}
}  // namespace testing