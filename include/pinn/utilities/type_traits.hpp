#pragma once

#include <utility>

namespace utilities {
template <class S1, class S2>
struct concat_int_sequence;

template <class T, T... Is1, T... Is2>
struct concat_int_sequence<std::integer_sequence<T, Is1...>,
                           std::integer_sequence<T, Is2...>> {
  using type = std::integer_sequence<T, Is1..., Is2...>;
};

template <class S1, class S2>
using concat_int_sequence_t = typename concat_int_sequence<S1, S2>::type;
}  // namespace utilities