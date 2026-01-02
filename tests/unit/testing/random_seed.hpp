#pragma once

#include <cstdint>

#include <catch2/catch_get_random_seed.hpp>

namespace testing {
inline std::uint32_t random_seed() { return Catch::getSeed(); }
}  // namespace testing