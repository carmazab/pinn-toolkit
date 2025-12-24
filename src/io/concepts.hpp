#pragma once

#include <concepts>
#include <string>

#include "core/concepts.hpp"

namespace io {
template <typename O>
concept OptionLike = requires(O o) {
  typename O::type;
  { o.value } -> std::same_as<typename O::type&>;
  { O::name() } -> std::same_as<std::string>;
};

template <typename W, typename S>
concept WriterLike =
    core::StateLike<S> && requires(W w, typename S::data_t t, const S& s) {
      { w.append(t, s) } -> std::same_as<void>;
    };

template <class T>
concept H5WritableState = core::StateLike<T> && requires(T s) {
  { s.data() } -> std::same_as<typename T::data_t*>;
} && std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>;

}  // namespace io