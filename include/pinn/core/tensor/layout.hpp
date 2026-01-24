#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <utility>

#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
namespace layout_detail {
template <index_t... Permutation>
consteval bool has_unique_elements() {
  std::array<index_t, sizeof...(Permutation)> permutation{Permutation...};
  std::sort(permutation.begin(), permutation.end());
  for (index_t i{1}; i < sizeof...(Permutation); ++i) {
    if (permutation[i] == permutation[i - 1]) {
      return false;
    }
  }
  return true;
}

template <index_t... Permutation>
concept NonEmpty = (sizeof...(Permutation) > 0);

template <index_t... Permutation>
concept InRange = ((Permutation < sizeof...(Permutation)) && ...);

template <index_t... Permutation>
concept UniqueElements = has_unique_elements<Permutation...>();
}  // namespace layout_detail

template <index_t... Permutation>
concept Valid = layout_detail::NonEmpty<Permutation...> &&
                layout_detail::InRange<Permutation...> &&
                layout_detail::UniqueElements<Permutation...>;

template <index_t... Permutation>
  requires Valid<Permutation...>
struct Layout {
  using indexer_t = std::array<index_t, sizeof...(Permutation)>;
  using permutation_t = std::integer_sequence<index_t, Permutation...>;

  static constexpr index_t rank = sizeof...(Permutation);
  static constexpr indexer_t permutation = {Permutation...};

  template <class... Extents>
    requires(sizeof...(Extents) == rank)
  static constexpr indexer_t strides_from_extents(Extents... extents) noexcept {
    return strides_from_extents(indexer_t{static_cast<index_t>(extents)...});
  }

  static constexpr indexer_t strides_from_extents(indexer_t extents) noexcept {
    indexer_t result{};
    result[permutation[0]] = 1;
    for (index_t j{0}; j < extents.size() - 1; ++j) {
      result[permutation[j + 1]] =
          extents[permutation[j]] * result[permutation[j]];
    }
    return result;
  }
};
}  // namespace tensor
}  // namespace core