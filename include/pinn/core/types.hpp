#pragma once

#include <array>
#include <cstddef>

namespace core {
using index_t = std::size_t;

template <std::size_t N>
using indices_t = std::array<index_t, N>;
}  // namespace core