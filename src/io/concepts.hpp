#pragma once

#include <concepts>

#include "core/concepts.hpp"

namespace io {
template <typename W, typename S>
concept WriterLike = core::StateLike<S> && requires(W w, double t, const S& s) {
  { w.append(t, s) } -> std::same_as<void>;
};
}  // namespace io