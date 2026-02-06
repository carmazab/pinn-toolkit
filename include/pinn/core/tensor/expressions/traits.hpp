#pragma once

#include "pinn/core/tensor/expressions/expressions.hpp"
#include "pinn/core/tensor/view.hpp"
#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
namespace expressions {
template <class Expr>
struct data_type;

template <class DataT>
struct data_type<Scalar<DataT>> {
  using type = DataT;
};

template <class Op, class Expr>
struct data_type<Unary<Op, Expr>> {
  using type = typename data_type<Expr>::type;
};

template <class DataT, class LayoutT>
struct data_type<View<DataT, LayoutT>> {
  using type = DataT;
};

template <class Expr>
using data_type_t = typename data_type<Expr>::type;

template <class Expr>
struct extents;

template <class DataT>
struct extents<Scalar<DataT>> {
  static constexpr auto value(const Scalar<DataT>&) noexcept {
    return indices_t<0>{};
  }
};

template <class Op, class Expr>
struct extents<Unary<Op, Expr>> {
  static constexpr auto value(const Expr& expr) noexcept {
    return extents<Expr>::value(expr);
  }
};

template <class DataT, class LayoutT>
struct extents<View<DataT, LayoutT>> {
  static constexpr auto value(const View<DataT, LayoutT>& view) noexcept {
    return view.extents;
  }
};

template <class Expr>
struct rank;

template <class DataT>
struct rank<Scalar<DataT>> {
  static constexpr index_t value = 0;
};

template <class Op, class Expr>
struct rank<Unary<Op, Expr>> {
  static constexpr index_t value = rank<Expr>::value;
};

template <class DataT, class LayoutT>
struct rank<View<DataT, LayoutT>> {
  static constexpr index_t value = LayoutT::rank;
};

template <class Expr>
constexpr index_t rank_v = rank<Expr>::value;
}  // namespace expressions
}  // namespace tensor
}  // namespace core