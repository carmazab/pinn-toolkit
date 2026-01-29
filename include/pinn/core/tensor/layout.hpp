#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <utility>

#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
namespace layout_detail {
template <index_t... Elements>
consteval bool is_axis_permutation_v() {
  constexpr auto size{sizeof...(Elements)};
  if (size == 0) {
    return false;
  }

  std::array<index_t, size> elements{Elements...};
  std::sort(elements.begin(), elements.end());
  for (index_t j{0}; j < size; ++j) {
    if (elements[j] != j) {
      return false;
    }
  }

  return true;
}
}  // namespace layout_detail

template <index_t... Elements>
concept AxisPermutation = layout_detail::is_axis_permutation_v<Elements...>();

template <index_t... StrideOrder>
  requires AxisPermutation<StrideOrder...>
struct Layout {
  using indexer_t = std::array<index_t, sizeof...(StrideOrder)>;
  using stride_order_t = std::integer_sequence<index_t, StrideOrder...>;

  static constexpr index_t rank = sizeof...(StrideOrder);
  static constexpr indexer_t stride_order = {StrideOrder...};

  template <class... Extents>
    requires(sizeof...(Extents) == rank)
  static constexpr indexer_t strides_from_extents(Extents... extents) noexcept {
    return strides_from_extents(indexer_t{static_cast<index_t>(extents)...});
  }

  static constexpr indexer_t strides_from_extents(indexer_t extents) noexcept {
    indexer_t result{};
    result[stride_order[0]] = 1;
    for (index_t j{0}; j < extents.size() - 1; ++j) {
      result[stride_order[j + 1]] =
          extents[stride_order[j]] * result[stride_order[j]];
    }
    return result;
  }
};

template <class StrideOrderT>
struct make_layout;

template <index_t... StrideOrder>
struct make_layout<std::integer_sequence<index_t, StrideOrder...>> {
  using type = Layout<StrideOrder...>;
};

template <class StrideOrderT>
using make_layout_t = typename make_layout<StrideOrderT>::type;
}  // namespace tensor
}  // namespace core