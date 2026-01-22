#pragma once

#include <type_traits>
#include <utility>

#include "pinn/core/assert.hpp"
#include "pinn/core/tensor/buffer.hpp"
#include "pinn/core/tensor/layout.hpp"
#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
namespace view_detail {
template <class S1, class S2>
struct integer_sequence_cat;

template <class T, T... Is1, T... Is2>
struct integer_sequence_cat<std::integer_sequence<T, Is1...>,
                            std::integer_sequence<T, Is2...>> {
  using type = std::integer_sequence<T, Is1..., Is2...>;
};

template <class T>
struct flag {
  static constexpr T value = 999;
};

template <class T, T Dim, class S>
struct flag_dim;

template <class T, T Dim, T... Is>
struct flag_dim<T, Dim, std::integer_sequence<T, Is...>> {
  using type = std::integer_sequence<T, (Is < Dim   ? Is
                                         : Is > Dim ? Is - 1
                                                    : flag<T>::value)...>;
};

template <class S>
struct remove_flagged;

template <class T, T Single>
struct remove_flagged<std::integer_sequence<T, Single>> {
  using type =
      std::conditional_t<Single == flag<T>::value, std::integer_sequence<T>,
                         std::integer_sequence<T, Single>>;
};

template <class T, T First, T... Rest>
struct remove_flagged<std::integer_sequence<T, First, Rest...>> {
  using rest = typename remove_flagged<std::integer_sequence<T, Rest...>>::type;
  using type =
      std::conditional_t<First == flag<T>::value, rest,
                         typename integer_sequence_cat<
                             std::integer_sequence<T, First>, rest>::type>;
};

template <class T, T Pivot, class S>
struct sliced_order;

template <class T, T Pivot, T... Dims>
struct sliced_order<T, Pivot, std::integer_sequence<T, Dims...>> {
  using type = typename remove_flagged<typename flag_dim<
      T, Pivot, std::integer_sequence<T, Dims...>>::type>::type;
};

template <class S>
struct make_layout;

template <index_t... Order>
struct make_layout<std::integer_sequence<index_t, Order...>> {
  using type = Layout<Order...>;
};

template <index_t Pivot, class S>
struct sliced_layout_impl;

template <index_t Pivot, index_t... Dims>
struct sliced_layout_impl<Pivot, std::integer_sequence<index_t, Dims...>> {
  using type = typename make_layout<typename view_detail::sliced_order<
      index_t, Pivot, std::integer_sequence<index_t, Dims...>>::type>::type;
};
}  // namespace view_detail

template <index_t Sliced, class LayoutT>
  requires(Sliced < LayoutT::rank)
using sliced_layout =
    typename view_detail::sliced_layout_impl<Sliced,
                                             typename LayoutT::ordering>::type;

template <class LayoutT, class DataT>
struct View {
  using layout_t = LayoutT;
  using indexer_t = layout_t::indexer_t;
  using data_t = DataT;

  static constexpr index_t rank = layout_t::rank;

  data_t* data;
  indexer_t extents;
  indexer_t strides;

  template <class... Indices>
    requires(sizeof...(Indices) == rank)
  constexpr data_t& operator()(Indices... indices) const noexcept {
    return operator()(indexer_t{static_cast<index_t>(indices)...});
  }

  constexpr data_t& operator()(const indexer_t& index) const noexcept {
#ifndef NDEBUG
    for (index_t j{0}; j < rank; ++j) {
      CORE_ASSERT(index[j] < extents[j], "View index out of bounds");
    }
#endif

    index_t offset{0};
    for (index_t j{0}; j < rank; ++j) {
      offset += index[j] * strides[j];
    }

    return data[offset];
  }

  template <index_t Dim>
  constexpr View<sliced_layout<Dim, layout_t>, data_t> slice(
      index_t index) const noexcept {
    static_assert(rank > 1, "Slicing a tensor of rank less than 2 not allowed");
    static_assert(Dim < rank, "Slice dimension out of bounds");

#ifndef NDEBUG
    CORE_ASSERT(index < extents[Dim], "Slice index out of bounds");
#endif

    std::array<index_t, rank - 1> subextents;
    std::array<index_t, rank - 1> substrides;

    index_t idx{0};
    for (index_t r{0}; r < rank; ++r) {
      if (r != Dim) {
        subextents[idx] = extents[r];
        substrides[idx] = strides[r];
        ++idx;
      }
    }

    return View<sliced_layout<Dim, layout_t>, data_t>{
        data + index * strides[Dim], subextents, substrides};
  }
};

template <class LayoutT, class DataT>
constexpr View<LayoutT, DataT> make_view(
    DataT* data, const typename LayoutT::indexer_t& extents) noexcept {
  View<LayoutT, DataT> view{};
  view.data = data;
  view.extents = extents;
  view.strides = LayoutT::strides_from_extents(extents);
  return view;
}

template <class LayoutT, class DataT>
constexpr View<LayoutT, DataT> make_view(
    Buffer<DataT>& buffer,
    const typename LayoutT::indexer_t& extents) noexcept {
  return View<LayoutT, DataT>{buffer.data(), extents,
                              LayoutT::strides_from_extents(extents)};
}

template <class LayoutT, class DataT>
constexpr View<LayoutT, const DataT> make_view(
    const Buffer<DataT>& buffer,
    const typename LayoutT::indexer_t& extents) noexcept {
  return View<LayoutT, const DataT>{buffer.data(), extents,
                                    LayoutT::strides_from_extents(extents)};
}
}  // namespace tensor
}  // namespace core