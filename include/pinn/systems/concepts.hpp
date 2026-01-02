#pragma once

#include <concepts>

#include "pinn/core/concepts.hpp"
#include "pinn/core/types.hpp"

namespace systems {
namespace concepts {
template <class S>
concept System =
    requires(S s, typename S::data_t time, const typename S::state_t& state) {
      typename S::state_t;
      requires core::concepts::State<typename S::state_t>;
      requires std::same_as<typename S::data_t, typename S::state_t::data_t>;

      { S::dof } -> std::convertible_to<core::index_t>;

      { s.dudt(time, state) } -> std::same_as<typename S::state_t>;
    };

template <class H>
concept Hamiltonian = System<H> && requires(H h, typename H::data_t time,
                                            const typename H::state_t& state) {
  { h.hamiltonian(time, state) } -> std::same_as<typename H::data_t>;
};
}  // namespace concepts
}  // namespace systems