#include "io/h5_writer.hpp"
#include "io/observer.hpp"
#include "numalg/timesteppers/runge_kutta_4.hpp"
#include "systems/ode/lotka_volterra.hpp"

int main() {
  using system_t = systems::ode::LotkaVolterra;
  using state_t = system_t::state_t;

  const system_t system{0.05, 0.01, 0.03, 0.005};

  double time{0.0};
  system_t::state_t state{0.5, 0.5};

  const double step_size{0.1};
  numalg::timesteppers::RungeKutta4<system_t> stepper{step_size};

  io::H5Writer<state_t> writer{"lotka_volterra.h5"};
  io::Observer observer{writer, 1};

  for (int step{0}; step < 10'000; ++step) {
    state = stepper.update(time, state, system);
    time += step_size;

    observer(time, state, step);
  }
}