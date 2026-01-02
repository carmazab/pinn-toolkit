#pragma once

#include <tuple>
#include <variant>

#include "pinn/io/concepts.hpp"
#include "pinn/numerics/concepts.hpp"
#include "pinn/numerics/timesteppers/runge_kutta_4.hpp"
#include "pinn/systems/concepts.hpp"

namespace numerics {
namespace timesteppers {
template <concepts::TimeStepper StepperT>
struct Traits;

template <systems::concepts::System SystemT>
struct Traits<RungeKutta4<SystemT>> {
  using params_t = RungeKutta4<SystemT>::Parameters;

  template <io::concepts::Node NodeT>
  static RungeKutta4<SystemT>::Parameters read_parameters(const NodeT& params) {
    return params_t{params[params_t::StepSize::name()]
                        .template as<typename params_t::StepSize::type>()};
  }
};

template <systems::concepts::System SystemT>
using all_steppers = std::tuple<RungeKutta4<SystemT>>;

template <systems::concepts::System SystemT>
using any_stepper = std::variant<RungeKutta4<SystemT>>;
}  // namespace timesteppers
}  // namespace numerics