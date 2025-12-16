#pragma once

#include <concepts>

#include "core/types.hpp"

namespace core {
template <class T>
concept RealLike = std::floating_point<T>;

template <typename S>
concept StateLike = requires(S s, const S cs, index_t i) {
  { S::dof } -> std::convertible_to<index_t>;

  typename S::data_t;
  requires RealLike<typename S::data_t>;
  { s[i] } -> std::same_as<typename S::data_t&>;
  { cs[i] } -> std::same_as<const typename S::data_t&>;
};
}  // namespace core