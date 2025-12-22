#pragma once

#include <array>

#include "core/concepts.hpp"
#include "core/types.hpp"

namespace core {
namespace detail {
template <StateLike StateT>
constexpr void add(StateT& lhs, const StateT& rhs) noexcept {
  for (index_t j{0}; j < StateT::dof; ++j) {
    lhs[j] += rhs[j];
  }
}

template <StateLike StateT>
constexpr StateT add(const StateT& lhs, const StateT& rhs) noexcept {
  StateT out{lhs};
  add(out, rhs);
  return out;
}

template <StateLike StateT>
constexpr void multiply(StateT& lhs, typename StateT::data_t scalar) noexcept {
  for (index_t j{0}; j < StateT::dof; ++j) {
    lhs[j] *= scalar;
  }
}

template <StateLike StateT>
constexpr StateT multiply(const StateT& lhs,
                          typename StateT::data_t scalar) noexcept {
  StateT out{lhs};
  multiply(out, scalar);
  return out;
}
}  // namespace detail

template <index_t Dof>
struct State {
  using data_t = double;

  static constexpr index_t dof = Dof;

  template <typename... Vars>
  constexpr explicit State(Vars... vars) noexcept
    requires(sizeof...(Vars) == Dof)
      : data_{vars...} {}

  constexpr data_t& operator[](index_t index) noexcept { return data_[index]; }
  constexpr const data_t& operator[](index_t index) const noexcept {
    return data_[index];
  }

  constexpr double* data() noexcept { return data_.data(); }

  constexpr const double* data() const noexcept { return data_.data(); }

 protected:
  std::array<data_t, Dof> data_;
};

template <StateLike StateT>
constexpr StateT operator+(const StateT& lhs, const StateT& rhs) noexcept {
  return detail::add(lhs, rhs);
}

template <StateLike StateT>
constexpr StateT operator*(const StateT& lhs,
                           typename StateT::data_t scalar) noexcept {
  return detail::multiply(lhs, scalar);
}

template <StateLike StateT>
constexpr StateT operator*(typename StateT::data_t scalar,
                           const StateT& rhs) noexcept {
  return detail::multiply(rhs, scalar);
}
}  // namespace core