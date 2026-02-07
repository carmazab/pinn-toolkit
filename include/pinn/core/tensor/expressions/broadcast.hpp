#pragma once

#include <algorithm>
#include <stdexcept>

#include "pinn/core/tensor/expressions/traits.hpp"
#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
namespace expressions {
namespace detail {
struct broadcast_error : std::logic_error {
  using std::logic_error::logic_error;
};

template <index_t rank, class Lhs, class Rhs>
auto broadcast_extents_impl(const Lhs& lhs, const Rhs& rhs) {
  static_assert(rank_v<Lhs> > 0 && rank_v<Rhs> > 0,
                "zero-rank operands in non-zero extents implementation");

  const auto exts_l{extents<Lhs>::value(lhs)};
  const auto exts_r{extents<Rhs>::value(rhs)};

  int jl{rank_v<Lhs> - 1};
  int jr{rank_v<Rhs> - 1};

  int k{rank - 1};
  indices_t<rank> result{};

  while (jl >= 0 and jr >= 0) {
    index_t el{exts_l[jl--]};
    index_t er{exts_r[jr--]};

    if (el == 1) {
      result[k--] = er;
    } else if (er == 1 or er == el) {
      result[k--] = el;
    } else {
      throw broadcast_error("operands shapes not compatible for broadcasting");
    }
  }

  while (jl >= 0) {
    result[k--] = exts_l[jl--];
  }

  while (jr >= 0) {
    result[k--] = exts_r[jr--];
  }

  return result;
}
}  // namespace detail

template <class Lhs, class Rhs>
struct broadcast_result {
  static constexpr index_t broadcast_rank = std::max(rank_v<Lhs>, rank_v<Rhs>);

  static auto broadcast_extents(const Lhs& lhs, const Rhs& rhs) {
    if constexpr (rank_v<Lhs> == 0) {
      return extents<Rhs>::value(rhs);
    } else if constexpr (rank_v<Rhs> == 0) {
      return extents<Lhs>::value(lhs);
    } else {
      return detail::broadcast_extents_impl<broadcast_rank>(lhs, rhs);
    }
  }
};
}  // namespace expressions
}  // namespace tensor
}  // namespace core