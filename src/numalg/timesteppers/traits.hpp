#pragma once

#include <tuple>
#include <variant>

#include "numalg/timesteppers/concepts.hpp"
#include "numalg/timesteppers/runge_kutta_4.hpp"
#include "systems/concepts.hpp"

namespace numalg {
namespace timesteppers {
template <concepts::TimeStepper StepperT>
struct Traits;

template <systems::concepts::System SystemT>
struct Traits<RungeKutta4<SystemT>> {
  using params_t = RungeKutta4<SystemT>::Parameters;

  template <class Node>
  static RungeKutta4<SystemT>::Parameters read_parameters(const Node& params) {
    return params_t{params[params_t::StepSize::name()]
                        .as<typename params_t::StepSize::type>()};
  }
};

template <systems::concepts::System SystemT>
using all_steppers = std::tuple<RungeKutta4<SystemT>>;

template <systems::concepts::System SystemT>
using any_stepper = std::variant<RungeKutta4<SystemT>>;
}  // namespace timesteppers
}  // namespace numalg