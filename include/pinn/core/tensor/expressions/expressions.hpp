#pragma once

namespace core {
namespace tensor {
namespace expressions {
template <class Op, class Lhs, class Rhs>
struct Binary {
  Lhs lhs;
  Rhs rhs;
};

template <class DataT>
struct Scalar {
  DataT value;
};

template <class Op, class Expr>
struct Unary {
  Expr expr;
};
}  // namespace expressions
}  // namespace tensor
}  // namespace core