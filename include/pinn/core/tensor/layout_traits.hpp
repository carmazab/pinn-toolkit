#pragma once

#include <type_traits>
#include <utility>

#include "pinn/utilities/type_traits.hpp"

namespace core {
namespace tensor {
namespace detail {
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

template <class T, T Last>
struct remove_flagged<std::integer_sequence<T, Last>> {
  using type =
      std::conditional_t<Last == flag<T>::value, std::integer_sequence<T>,
                         std::integer_sequence<T, Last>>;
};

template <class T, T First, T... Rest>
struct remove_flagged<std::integer_sequence<T, First, Rest...>> {
  using rest = typename remove_flagged<std::integer_sequence<T, Rest...>>::type;
  using type = std::conditional_t<
      First == flag<T>::value, rest,
      utilities::concat_int_sequence_t<std::integer_sequence<T, First>, rest>>;
};

template <class T, T Sliced, class S>
struct sliced_order;

template <class T, T ToSlice, T... Dims>
struct sliced_order<T, ToSlice, std::integer_sequence<T, Dims...>> {
  using type = typename remove_flagged<typename flag_dim<
      T, ToSlice, std::integer_sequence<T, Dims...>>::type>::type;
};

template <class S>
struct make_layout;

template <class T, T... Permutation>
struct make_layout<std::integer_sequence<T, Permutation...>> {
  using type = Layout<Permutation...>;
};

template <class T, T ToSlice, class S>
struct sliced_layout_impl;

template <class T, T ToSlice, T... Dims>
struct sliced_layout_impl<T, ToSlice, std::integer_sequence<T, Dims...>> {
  using type = typename make_layout<typename sliced_order<
      T, ToSlice, std::integer_sequence<T, Dims...>>::type>::type;
};
}  // namespace detail

template <class T, T ToSlice, class LayoutT>
  requires(ToSlice < LayoutT::rank)
using sliced_layout =
    typename detail::sliced_layout_impl<T, ToSlice,
                                        typename LayoutT::ordering>::type;
}  // namespace tensor
}  // namespace core