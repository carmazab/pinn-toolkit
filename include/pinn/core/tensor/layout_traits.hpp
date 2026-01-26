#pragma once

#include <array>
#include <utility>

#include "pinn/core/tensor/layout.hpp"

namespace core {
namespace tensor {
namespace layout_detail {
template <class T, T Dim, T... Permutation>
consteval auto slice_permutation_array() {
  constexpr T flag{999};

  constexpr auto rank{sizeof...(Permutation)};
  std::array<T, rank> tmp{Permutation...};
  for (index_t i{0}; i < rank; ++i) {
    tmp[i] = tmp[i] < Dim ? tmp[i] : tmp[i] > Dim ? tmp[i] - 1 : flag;
  }

  index_t j{0};
  std::array<T, rank - 1> result{};
  for (index_t i{0}; i < rank; ++i) {
    if (tmp[i] != flag) {
      result[j++] = tmp[i];
    }
  }

  return result;
}

template <class T, T Dim, T... Permutation>
consteval auto slice_permutation() {
  constexpr auto perm{slice_permutation_array<T, Dim, Permutation...>()};
  return [&]<index_t... Is>(std::integer_sequence<index_t, Is...>) {
    return std::integer_sequence<T, perm[Is]...>{};
  }(std::make_integer_sequence<index_t, perm.size()>{});
}

template <class T, T Dim, class S>
struct sliced_permutation;

template <class T, T Dim, T... Permutation>
struct sliced_permutation<T, Dim, std::integer_sequence<T, Permutation...>> {
  using type = decltype(slice_permutation<T, Dim, Permutation...>());
};
}  // namespace layout_detail

template <class S>
struct make_layout;

template <class T, T... Permutation>
struct make_layout<std::integer_sequence<T, Permutation...>> {
  using type = Layout<Permutation...>;
};

template <class T, T Dim, class LayoutT>
using sliced_layout =
    typename make_layout<typename layout_detail::sliced_permutation<
        T, Dim, typename LayoutT::permutation_t>::type>::type;
}  // namespace tensor
}  // namespace core