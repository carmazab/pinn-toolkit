#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <utility>

#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
namespace layout_detail {
template <index_t... Sequence>
consteval bool is_unique_v() {
  std::array<index_t, sizeof...(Sequence)> seq{Sequence...};
  std::sort(seq.begin(), seq.end());
  for (index_t i{1}; i < sizeof...(Sequence); ++i) {
    if (seq[i] == seq[i - 1]) {
      return false;
    }
  }
  return true;
}
template <index_t... Sequence>
concept NonEmpty = (sizeof...(Sequence) > 0);

template <index_t... Sequence>
concept InRange = ((Sequence < sizeof...(Sequence)) && ...);

template <index_t... Sequence>
concept Unique = layout_detail::is_unique_v<Sequence...>();
}  // namespace layout_detail

template <index_t... Sequence>
concept LayoutOrder =
    layout_detail::NonEmpty<Sequence...> &&
    layout_detail::InRange<Sequence...> && layout_detail::Unique<Sequence...>;

template <index_t... Order>
  requires LayoutOrder<Order...>
struct Layout {
  using ordering = std::integer_sequence<index_t, Order...>;
  using indexer_t = std::array<index_t, sizeof...(Order)>;

  static constexpr indexer_t order = {Order...};
  static constexpr index_t rank = sizeof...(Order);

  template <class... Extents>
    requires(sizeof...(Extents) == rank)
  static constexpr indexer_t strides_from_extents(Extents... extents) noexcept {
    return strides_from_extents(indexer_t{static_cast<index_t>(extents)...});
  }

  static constexpr indexer_t strides_from_extents(indexer_t extents) noexcept {
    indexer_t result{};
    result[order[0]] = 1;
    for (index_t j{0}; j < extents.size() - 1; ++j) {
      result[order[j + 1]] = extents[order[j]] * result[order[j]];
    }
    return result;
  }
};
}  // namespace tensor
}  // namespace core