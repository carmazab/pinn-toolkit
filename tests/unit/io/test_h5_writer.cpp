#include <string>
#include <vector>

#include "pinn/core/random/rng_state.hpp"
#include "pinn/core/state.hpp"
#include "pinn/core/types.hpp"
#include "pinn/io/h5_writer.hpp"
#include "testing/check_equal.hpp"
#include "testing/random_seed.hpp"

namespace {
void run_test_suite() {
  using state_t = core::State<4>;
  using data_t = state_t::data_t;

  constexpr core::index_t dof{state_t::dof};

  core::random::RngState rng{testing::random_seed()};

  const data_t mean{0.0}, stddev{3.0};
  const std::vector<data_t> times{
      rng.normal(mean, stddev), rng.normal(mean, stddev),
      rng.normal(mean, stddev), rng.normal(mean, stddev),
      rng.normal(mean, stddev)};
  const std::vector<state_t> states{
      state_t{rng.normal(mean, stddev), rng.normal(mean, stddev),
              rng.normal(mean, stddev), rng.normal(mean, stddev)},
      state_t{rng.normal(mean, stddev), rng.normal(mean, stddev),
              rng.normal(mean, stddev), rng.normal(mean, stddev)},
      state_t{rng.normal(mean, stddev), rng.normal(mean, stddev),
              rng.normal(mean, stddev), rng.normal(mean, stddev)},
      state_t{rng.normal(mean, stddev), rng.normal(mean, stddev),
              rng.normal(mean, stddev), rng.normal(mean, stddev)},
      state_t{rng.normal(mean, stddev), rng.normal(mean, stddev),
              rng.normal(mean, stddev), rng.normal(mean, stddev)}};

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

  std::vector<data_t> time_values_buffer(size);
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

  std::vector<data_t> state_values_buffer(size * dof);
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