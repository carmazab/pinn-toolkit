#pragma once

#include "core/concepts.hpp"
#include "core/types.hpp"
#include "framework/framework.hpp"

namespace testing {
template <class T>
void check_equal(T a, T b) {
  CHECK(a == b);
}

template <core::RealLike R>
void check_equal_within(R a, R b, R rel_tol = static_cast<R>(1.0e-12),
                        R abs_tol = static_cast<R>(1.0e-14)) {
  CHECK_THAT(a, Catch::Matchers::WithinRel(b, rel_tol) ||
                    Catch::Matchers::WithinAbs(b, abs_tol));
}

template <core::StateLike A>
void check_equal_within(
    const A& a, const A& b,
    typename A::data_t tol = static_cast<typename A::data_t>(1.0e-12)) {
  for (core::index_t j{0}; j < A::dof; ++j) {
    check_equal_within(a[j], b[j]);
  }
}
}  // namespace testing