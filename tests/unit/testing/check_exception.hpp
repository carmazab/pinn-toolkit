#pragma once

#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

namespace testing {
template <class Exception, class F>
void check_throws_as(F&& f) {
  CHECK_THROWS_AS(f(), Exception);
}

template <class F>
void check_throws_as_logic_error(F&& f) {
  check_throws_as<std::logic_error>(f);
}
}  // namespace testing