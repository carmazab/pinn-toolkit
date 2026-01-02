#pragma once

#include <concepts>
#include <string>

#include "pinn/systems/concepts.hpp"

namespace numerics {
namespace concepts {
template <class T>
concept TimeStepper =
    requires(T t, typename T::data_t time, const typename T::state_t& state,
             const typename T::system_t& system) {
      typename T::system_t;
      requires systems::concepts::System<typename T::system_t>;
      requires std::same_as<typename T::state_t, typename T::system_t::state_t>;
      requires std::same_as<typename T::data_t, typename T::state_t::data_t>;

      typename T::Parameters;

      { T::name() } -> std::same_as<std::string>;

      { t.update(time, state, system) } -> std::same_as<typename T::state_t>;
    };
}  // namespace concepts
}  // namespace numerics