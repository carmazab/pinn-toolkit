#pragma once

#include <H5Cpp.h>

#include <string>

#include "io/concepts.hpp"
#include "io/h5_error_stack_silencer.hpp"

namespace io {
template <H5WritableState StateT>
struct H5Writer {
  static constexpr hsize_t time_rank = 1;
  static constexpr hsize_t state_rank = 2;

  explicit H5Writer(const std::string& filename, hsize_t chunk_length = 1024)
      : file_{filename, H5F_ACC_TRUNC},
        chunk_length_{chunk_length},
        capacity_{0},
        size_{0},
        finalized_{false} {
    create_time_dataset();
    create_state_dataset();

    hsize_t time_count[time_rank]{1};
    time_memspace_ = H5::DataSpace{time_rank, time_count};

    hsize_t state_count[state_rank]{1, StateT::dof};
    state_memspace_ = H5::DataSpace{state_rank, state_count};
  }

  ~H5Writer() {
    try {
      close();
    } catch (...) {
    }
  }

  const hsize_t size() const { return size_; }

  void append(double time, const StateT& state) {
    extend_data_if_needed(512);
    write_time(time);
    write_state(state);
    ++size_;
  }

  void close() {
    if (!finalized_) {
      hsize_t time_size[1] = {size_};
      time_ds_.extend(time_size);

      hsize_t state_size[2] = {size_, StateT::dof};
      state_ds_.extend(state_size);

      file_.flush(H5F_SCOPE_GLOBAL);
      finalized_ = true;
    }
  }

 private:
  void create_time_dataset() {
    const std::string name{"time"};
    try {
      // suppress expected output when opening dataset fails
      HDF5ErrorStackSilencer silent;
      time_ds_ = file_.openDataSet(name);

    } catch (const H5::Exception&) {
      hsize_t dims[time_rank]{0};
      hsize_t max_dims[time_rank]{H5S_UNLIMITED};
      H5::DataSpace space{time_rank, dims, max_dims};

      H5::DSetCreatPropList plist;
      hsize_t chunk[time_rank]{chunk_length_};
      plist.setChunk(time_rank, chunk);

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
      hsize_t dims[state_rank]{0, 0};
      hsize_t max_dims[state_rank]{H5S_UNLIMITED, StateT::dof};
      H5::DataSpace space{state_rank, dims, max_dims};

      H5::DSetCreatPropList plist;
      hsize_t chunk[state_rank]{chunk_length_, StateT::dof};
      plist.setChunk(state_rank, chunk);

      state_ds_ =
          file_.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space, plist);
    }
  }

  void extend_data_if_needed(hsize_t extend_by) {
    if (size_ >= capacity_) {
      capacity_ += extend_by;

      hsize_t time_size[1]{capacity_};
      time_ds_.extend(time_size);

      hsize_t state_size[2]{capacity_, StateT::dof};
      state_ds_.extend(state_size);
    }
  }

  void write_time(double time) {
    H5::DataSpace filespace{time_ds_.getSpace()};
    hsize_t start[time_rank]{size_};
    hsize_t count[time_rank]{1};
    filespace.selectHyperslab(H5S_SELECT_SET, count, start);

    time_ds_.write(&time, H5::PredType::NATIVE_DOUBLE, time_memspace_,
                   filespace);
  }

  void write_state(const StateT& state) {
    H5::DataSpace filespace{state_ds_.getSpace()};
    hsize_t start[state_rank]{size_, 0};
    hsize_t count[state_rank]{1, StateT::dof};
    filespace.selectHyperslab(H5S_SELECT_SET, count, start);

    state_ds_.write(state.data(), H5::PredType::NATIVE_DOUBLE, state_memspace_,
                    filespace);
  }

  H5::H5File file_;
  H5::DataSet time_ds_;
  H5::DataSpace time_memspace_;
  H5::DataSet state_ds_;
  H5::DataSpace state_memspace_;

  hsize_t chunk_length_;
  hsize_t capacity_;
  hsize_t size_;

  bool finalized_;
};
}  // namespace io