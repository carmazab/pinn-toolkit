#pragma once

#include <array>

#include "pinn/core/tensor/buffer.hpp"
#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
template <class DataT, index_t Rank>
struct View {
  static_assert(Rank > 0);

  using data_t = DataT;
  using indexer_t = std::array<index_t, Rank>;

  data_t* data;
  indexer_t extents;
  indexer_t strides;

  template <class... Indices>
    requires(sizeof...(Indices) == Rank)
  constexpr data_t& operator()(Indices... indices) const noexcept {
    return operator()(indexer_t{static_cast<index_t>(indices)...});
  }

  constexpr data_t& operator()(const indexer_t& index) const noexcept {
    index_t offset{0};
    for (index_t j{0}; j < Rank; ++j) {
      offset += index[j] * strides[j];
    }
    return data[offset];
  }

  template <index_t Dim>
  constexpr View<DataT, Rank - 1> slice(index_t index) const noexcept {
    std::array<index_t, Rank - 1> subextents, substrides;

    index_t idx{0};
    for (index_t r{0}; r < Rank; ++r) {
      if (r != Dim) {
        subextents[idx] = extents[r];
        substrides[idx] = strides[r];
        ++idx;
      }
    }

    return View<DataT, Rank - 1>{data + index * strides[Dim], subextents,
                                 substrides};
  }
};

namespace detail {
template <class IndexerT>
constexpr IndexerT strides_from_extents(const IndexerT& extents) noexcept {
  IndexerT result;
  result[0] = 1;
  for (index_t j{0}; j < extents.size() - 1; ++j) {
    result[j + 1] = extents[j] * result[j];
  }
  return result;
}
}  // namespace detail

template <class DataT, index_t Rank>
constexpr View<DataT, Rank> make_view(
    DataT* data, const std::array<index_t, Rank>& extents) noexcept {
  View<DataT, Rank> view{};
  view.data = data;
  view.extents = extents;
  view.strides = detail::strides_from_extents(extents);
  return view;
}

template <class DataT, index_t Rank>
constexpr View<DataT, Rank> make_view(
    Buffer<DataT>& buffer, const std::array<index_t, Rank>& extents) noexcept {
  return View<DataT, Rank>{buffer.data(), extents,
                           detail::strides_from_extents(extents)};
}

template <class DataT, index_t Rank>
constexpr View<const DataT, Rank> make_view(
    const Buffer<DataT>& buffer,
    const std::array<index_t, Rank>& extents) noexcept {
  return View<const DataT, Rank>{buffer.data(), extents,
                                 detail::strides_from_extents(extents)};
}

}  // namespace tensor
}  // namespace core