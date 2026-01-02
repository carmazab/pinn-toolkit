#include <string>
#include <vector>

#include "pinn/core/state.hpp"
#include "pinn/core/types.hpp"
#include "pinn/io/h5_writer.hpp"
#include "testing/check_equal.hpp"

namespace {
void run_test_suite() {
  using state_t = core::State<4>;
  constexpr core::index_t dof{state_t::dof};

  const std::vector<double> times{0.1, 0.2, 0.3, 0.4, 0.5};
  const std::vector<state_t> states{
      state_t{-1.0, 22.0, 3.0, -4.0}, state_t{-5.0, 6.0, -9.0, 11.0},
      state_t{542.0, 6123.0, 312.0, 41.0}, state_t{21.0, -412.0, -15.0, 65.0},
      state_t{-0.14, -12.41, 1231.0, -0.99986}};

  const std::string filename{"h5_writer_test.h5"};
  io::H5Writer<state_t> writer{filename};
  for (core::index_t n{0}; n < times.size(); ++n) {
    writer.append(times[n], states[n]);
  }
  writer.close();

  const hsize_t size{writer.size()};

  H5::H5File file{filename, H5F_ACC_RDONLY};

  H5::DataSet time_ds{file.openDataSet("time")};

  hsize_t time_dims_buffer[1];
  H5::DataSpace time_space{time_ds.getSpace()};
  time_space.getSimpleExtentDims(time_dims_buffer);
  testing::check_equal(time_dims_buffer[0], size);

  std::vector<double> time_values_buffer(size);
  time_ds.read(time_values_buffer.data(), H5::PredType::NATIVE_DOUBLE);
  for (core::index_t n{0}; n < times.size(); ++n) {
    testing::check_equal_within(time_values_buffer[n], times[n]);
  }

  H5::DataSet state_ds{file.openDataSet("state")};

  hsize_t state_dims_buffer[2];
  H5::DataSpace state_space{state_ds.getSpace()};
  state_space.getSimpleExtentDims(state_dims_buffer);
  testing::check_equal(state_dims_buffer[0], size);
  testing::check_equal(state_dims_buffer[1], hsize_t{dof});

  std::vector<double> state_values_buffer(size * dof);
  state_ds.read(state_values_buffer.data(), H5::PredType::NATIVE_DOUBLE);
  for (core::index_t n{0}; n < times.size(); ++n) {
    for (core::index_t j{0}; j < dof; ++j) {
      testing::check_equal_within(state_values_buffer[n * dof + j],
                                  states[n][j]);
    }
  }
}
}  // namespace

TEST_CASE("unit.io.h5_writer", "[unit][io]") { run_test_suite(); }