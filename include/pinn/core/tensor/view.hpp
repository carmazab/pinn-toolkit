#pragma once

#include "pinn/core/assert.hpp"
#include "pinn/core/tensor/buffer.hpp"
#include "pinn/core/tensor/layout.hpp"
#include "pinn/core/tensor/layout_traits.hpp"
#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
template <class LayoutT, class DataT>
struct View {
  using layout_t = LayoutT;
  using indexer_t = layout_t::indexer_t;
  using data_t = DataT;

  static constexpr index_t rank = layout_t::rank;

  data_t* data;
  indexer_t extents;
  indexer_t strides;

  template <class... Indices>
    requires(sizeof...(Indices) == rank)
  constexpr data_t& operator()(Indices... indices) const noexcept {
    return operator()(indexer_t{static_cast<index_t>(indices)...});
  }

  constexpr data_t& operator()(const indexer_t& index) const noexcept {
#ifndef NDEBUG
    for (index_t j{0}; j < rank; ++j) {
      CORE_ASSERT(index[j] < extents[j], "View index out of bounds");
    }
#endif

    index_t offset{0};
    for (index_t j{0}; j < rank; ++j) {
      offset += index[j] * strides[j];
    }

    return data[offset];
  }

  template <index_t Dim>
  constexpr View<sliced_layout<index_t, Dim, layout_t>, data_t> slice(
      index_t index) const noexcept {
    static_assert(rank > 1, "Slicing a tensor of rank less than 2 not allowed");
    static_assert(Dim < rank, "Slice dimension out of bounds");

#ifndef NDEBUG
    CORE_ASSERT(index < extents[Dim], "Slice index out of bounds");
#endif

    std::array<index_t, rank - 1> subextents;
    std::array<index_t, rank - 1> substrides;

    index_t idx{0};
    for (index_t r{0}; r < rank; ++r) {
      if (r != Dim) {
        subextents[idx] = extents[r];
        substrides[idx] = strides[r];
        ++idx;
      }
    }

    return View<sliced_layout<index_t, Dim, layout_t>, data_t>{
        data + index * strides[Dim], subextents, substrides};
  }
};

template <class LayoutT, class DataT>
constexpr View<LayoutT, DataT> make_view(
    DataT* data, const typename LayoutT::indexer_t& extents) noexcept {
  View<LayoutT, DataT> view{};
  view.data = data;
  view.extents = extents;
  view.strides = LayoutT::strides_from_extents(extents);
  return view;
}

template <class LayoutT, class DataT>
constexpr View<LayoutT, DataT> make_view(
    Buffer<DataT>& buffer,
    const typename LayoutT::indexer_t& extents) noexcept {
  return View<LayoutT, DataT>{buffer.data(), extents,
                              LayoutT::strides_from_extents(extents)};
}

template <class LayoutT, class DataT>
constexpr View<LayoutT, const DataT> make_view(
    const Buffer<DataT>& buffer,
    const typename LayoutT::indexer_t& extents) noexcept {
  return View<LayoutT, const DataT>{buffer.data(), extents,
                                    LayoutT::strides_from_extents(extents)};
}
}  // namespace tensor
}  // namespace core