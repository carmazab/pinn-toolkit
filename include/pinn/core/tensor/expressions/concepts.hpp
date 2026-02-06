#pragma once

#include <concepts>
#include <type_traits>

#include "pinn/core/tensor/expressions/traits.hpp"

namespace core {
namespace tensor {
namespace expressions {
template <class A>
concept Arithmetic = std::is_arithmetic_v<A>;

template <class E>
concept Expression = requires(const E& ce) {
  typename data_type<E>::type;
  rank<E>::value;

  { extents<E>::value(ce) };
};
}  // namespace expressions
}  // namespace tensor
}  // namespace core