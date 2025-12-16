#include "numalg/timesteppers/runge_kutta_4.hpp"
#include "systems/lotka_volterra.hpp"

int main() {
  using system_t = systems::LotkaVolterra;

  const system_t system{0.05, 0.01, 0.03, 0.005};

  const double dt{0.1};
  numalg::timesteppers::RungeKutta4 stepper{dt};

  // double time{0.0};
  system_t::state_t state{0.5, 0.5};

  for (int j{0}; j < 10'000; ++j) {
    state = stepper.update(dt, state, system);
    // time += dt;
  }
}