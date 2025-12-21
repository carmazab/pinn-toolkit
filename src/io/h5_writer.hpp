#pragma once

#include <H5Cpp.h>

#include <string>

#include "core/concepts.hpp"
#include "io/h5_error_stack_silencer.hpp"

namespace io {
template <core::StateLike StateT>
struct H5Writer {
  explicit H5Writer(const std::string& filename)
      : file_{filename, H5F_ACC_TRUNC}, step_{0} {
    create_time_dataset();
    create_state_dataset();
  }

  void append(double time, const StateT& state) {
    write_time(time);
    write_state(state);
    ++step_;
  }

 private:
  void create_time_dataset() {
    const std::string name{"time"};

    try {
      // suppress expected output when opening dataset fails
      HDF5ErrorStackSilencer silent;
      time_ds_ = file_.openDataSet(name);

    } catch (const H5::Exception&) {
      const hsize_t rank{1};

      hsize_t dims[rank]{0};
      hsize_t max_dims[rank]{H5S_UNLIMITED};
      H5::DataSpace space{rank, dims, max_dims};

      H5::DSetCreatPropList plist;
      hsize_t chunk[rank]{1024};
      plist.setChunk(rank, chunk);

      time_ds_ =
          file_.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space, plist);
    }
  }

  void create_state_dataset() {
    const std::string name{"state"};

    try {
      // suppress expected output when opening dataset fails
      HDF5ErrorStackSilencer silent;
      state_ds_ = file_.openDataSet(name);

    } catch (const H5::Exception&) {
      const hsize_t rank{2};

      hsize_t dims[rank]{0, 0};
      hsize_t max_dims[rank]{H5S_UNLIMITED, StateT::dof};
      H5::DataSpace space{rank, dims, max_dims};

      H5::DSetCreatPropList plist;
      hsize_t chunk[rank]{1024, StateT::dof};
      plist.setChunk(rank, chunk);

      state_ds_ =
          file_.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space, plist);
    }
  }

  void write_time(double t) {
    const hsize_t rank{1};

    hsize_t new_size[rank]{step_ + 1};
    time_ds_.extend(new_size);

    H5::DataSpace filespace{time_ds_.getSpace()};
    hsize_t start[rank]{step_};
    hsize_t count[rank]{1};
    filespace.selectHyperslab(H5S_SELECT_SET, count, start);

    H5::DataSpace memspace{rank, count};
    time_ds_.write(&t, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
  }

  void write_state(const StateT& state) {
    const hsize_t rank{2};
    const hsize_t dof{StateT::dof};

    hsize_t new_size[rank]{step_ + 1, dof};
    state_ds_.extend(new_size);

    H5::DataSpace filespace{state_ds_.getSpace()};
    hsize_t start[rank]{step_, 0};
    hsize_t count[rank]{1, dof};
    filespace.selectHyperslab(H5S_SELECT_SET, count, start);

    H5::DataSpace memspace{rank, count};
    state_ds_.write(&state, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
  }

  H5::H5File file_;
  H5::DataSet time_ds_;
  H5::DataSet state_ds_;
  hsize_t step_;
};
}  // namespace io