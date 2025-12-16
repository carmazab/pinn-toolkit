#pragma once

#include "framework/framework.hpp"

namespace testing {
template <class Exception, class F>
void check_throws_as(F&& f) {
  CHECK_THROWS_AS(f(), Exception);
}
}  // namespace testing