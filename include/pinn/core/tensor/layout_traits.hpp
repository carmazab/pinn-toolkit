#pragma once

#include <array>
#include <utility>

#include "pinn/core/tensor/layout.hpp"
#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
namespace layout_detail {
template <class T, T... Order, T... Map>
consteval auto apply_axis_map(std::integer_sequence<T, Order...>,
                              std::integer_sequence<T, Map...>) {
  constexpr auto size{sizeof...(Order)};
  constexpr std::array<T, size> order{Order...};
  constexpr std::array<T, size> map{Map...};
  return [&]<index_t... Is>(std::integer_sequence<index_t, Is...>) {
    return std::integer_sequence<T, order[map[Is]]...>{};
  }(std::make_integer_sequence<index_t, size>{});
}

template <class OrderT, class MapT>
using reorder_axes = decltype(apply_axis_map(OrderT{}, MapT{}));
}  // namespace layout_detail

template <class LayoutT, index_t... Map>
using reordered_layout = make_layout_t<layout_detail::reorder_axes<
    typename LayoutT::stride_order_t, std::integer_sequence<index_t, Map...>>>;

namespace layout_detail {
template <class T, T Dim, T... Order>
consteval auto sliced_order_as_array() {
  constexpr T flag{std::numeric_limits<T>::max()};

  constexpr auto size{sizeof...(Order)};
  std::array<T, size> tmp{Order...};
  for (index_t j{0}; j < size; ++j) {
    tmp[j] = tmp[j] < Dim ? tmp[j] : tmp[j] > Dim ? tmp[j] - 1 : flag;
  }

  index_t idx{0};
  std::array<T, size - 1> result{};
  for (index_t j{0}; j < size; ++j) {
    if (tmp[j] != flag) {
      result[idx++] = tmp[j];
    }
  }

  return result;
}

template <class T, T Dim, T... Order>
consteval auto apply_slicing(std::integer_sequence<T, Order...>) {
  constexpr auto sliced_order{sliced_order_as_array<T, Dim, Order...>()};
  return [&]<index_t... Is>(std::integer_sequence<index_t, Is...>) {
    return std::integer_sequence<T, sliced_order[Is]...>{};
  }(std::make_integer_sequence<index_t, sizeof...(Order) - 1>{});
}

template <class T, T Dim, class StrideOrderT>
using slice_stride_order = decltype(apply_slicing<T, Dim>(StrideOrderT{}));
}  // namespace layout_detail

template <class LayoutT, index_t Dim>
using sliced_layout = make_layout_t<layout_detail::slice_stride_order<
    index_t, Dim, typename LayoutT::stride_order_t>>;
}  // namespace tensor
}  // namespace core