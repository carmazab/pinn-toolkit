#pragma once

#include <utility>

#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
template <class DataT>
struct Buffer {
  using data_t = DataT;

  Buffer() noexcept : size_{0}, data_{nullptr} {}

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept : size_{other.size_}, data_{other.data_} {
    other.size_ = 0;
    other.data_ = nullptr;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      Buffer tmp{std::move(other)};
      swap(tmp);
    }
    return *this;
  }

  ~Buffer() noexcept { delete[] data_; }

  explicit Buffer(index_t size)
      : size_{size}, data_{size > 0 ? new data_t[size] : nullptr} {}

  data_t* data() noexcept { return data_; }
  const data_t* data() const noexcept { return data_; }

  index_t size() const noexcept { return size_; }

  void swap(Buffer& other) noexcept {
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
  }

 private:
  index_t size_;
  data_t* data_;
};

template <class DataT>
void swap(Buffer<DataT>& lhs, Buffer<DataT>& rhs) noexcept {
  lhs.swap(rhs);
}
}  // namespace tensor
}  // namespace core