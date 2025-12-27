#pragma once

#include <concepts>

#include "core/types.hpp"

namespace core {
namespace concepts {
template <class R>
concept Real = std::floating_point<R>;

template <class S>
concept State = requires(S s, const S cs, index_t index) {
  typename S::data_t;
  requires Real<typename S::data_t>;

  { S::dof } -> std::convertible_to<index_t>;

  { s[index] } -> std::same_as<typename S::data_t&>;
  { cs[index] } -> std::same_as<const typename S::data_t&>;

  { s.data() } -> std::same_as<typename S::data_t*>;
  { cs.data() } -> std::same_as<const typename S::data_t*>;
};
}  // namespace concepts
}  // namespace core