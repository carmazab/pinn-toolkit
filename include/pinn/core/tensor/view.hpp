#pragma once

#include <utility>

#include "pinn/core/assert.hpp"
#include "pinn/core/tensor/buffer.hpp"
#include "pinn/core/tensor/layout.hpp"
#include "pinn/core/tensor/layout_traits.hpp"
#include "pinn/core/types.hpp"

namespace core {
namespace tensor {
template <class DataT, class LayoutT>
struct View {
  using data_t = DataT;
  using layout_t = LayoutT;
  using indexer_t = layout_t::indexer_t;

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
    for (index_t j{0}; j < rank; ++j) {
      CORE_ASSERT(index[j] < extents[j], "View index out of bounds");
    }

    index_t offset{0};
    for (index_t j{0}; j < rank; ++j) {
      offset += index[j] * strides[j];
    }

    return data[offset];
  }

  template <index_t... AxisMap>
    requires AxisPermutation<AxisMap...> && (sizeof...(AxisMap) == rank)
  constexpr auto permute() const noexcept {
    return View<data_t, reordered_layout<layout_t, AxisMap...>>{
        data, indexer_t{extents[AxisMap]...}, indexer_t{strides[AxisMap]...}};
  }

  constexpr auto reverse_layout() const noexcept {
    return [&]<index_t... Is>(std::integer_sequence<index_t, Is...>) {
      return permute<(sizeof...(Is) - 1 - Is)...>();
    }(std::make_integer_sequence<index_t, rank>{});
  }

  template <index_t L, index_t R>
    requires(L < rank) && (R < rank)
  constexpr auto transpose() const noexcept {
    return [&]<index_t... Is>(std::integer_sequence<index_t, Is...>) {
      return permute<(Is == L ? R : Is == R ? L : Is)...>();
    }(std::make_integer_sequence<index_t, rank>{});
  }

  template <index_t Dim>
    requires(Dim < rank)
  constexpr auto slice(index_t index) const noexcept {
    CORE_ASSERT(index < extents[Dim], "Slice index out of bounds");

    using sliced_layout_t = sliced_layout<layout_t, Dim>;
    using sliced_indexer_t = sliced_layout_t::indexer_t;

    sliced_indexer_t subextents;
    sliced_indexer_t substrides;

    index_t idx{0};
    for (index_t j{0}; j < rank; ++j) {
      if (j != Dim) {
        subextents[idx] = extents[j];
        substrides[idx] = strides[j];
        ++idx;
      }
    }

    return View<data_t, sliced_layout_t>{data + index * strides[Dim],
                                         subextents, substrides};
  }
};

template <class LayoutT, class DataT>
constexpr auto make_view(DataT* data,
                         const typename LayoutT::indexer_t& extents) noexcept {
  return View<DataT, LayoutT>{data, extents,
                              LayoutT::strides_from_extents(extents)};
}

template <class LayoutT, class DataT>
constexpr auto make_view(Buffer<DataT>& buffer,
                         const typename LayoutT::indexer_t& extents) noexcept {
  return View<DataT, LayoutT>{buffer.data(), extents,
                              LayoutT::strides_from_extents(extents)};
}

template <class LayoutT, class DataT>
constexpr auto make_view(const Buffer<DataT>& buffer,
                         const typename LayoutT::indexer_t& extents) noexcept {
  return View<const DataT, LayoutT>{buffer.data(), extents,
                                    LayoutT::strides_from_extents(extents)};
}

template <class DataT>
constexpr auto make_view(DataT* data) noexcept {
  return make_view<Layout<>, DataT>(data, Layout<>::indexer_t{});
}

template <class DataT>
constexpr auto make_view(Buffer<DataT>& buffer) noexcept {
  return make_view<DataT>(buffer.data());
}

template <class DataT>
constexpr auto make_view(const Buffer<DataT>& buffer) noexcept {
  return make_view<const DataT>(buffer.data());
}
}  // namespace tensor
}  // namespace core