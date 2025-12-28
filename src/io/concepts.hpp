#pragma once

#include <concepts>
#include <string>

#include "core/concepts.hpp"
#include "core/types.hpp"

namespace io {
namespace concepts {
template <class N>
concept Node = requires(N n, const std::string& key) {
  { n[key] };
};

template <class O>
concept Option = requires(O o) {
  typename O::type;
  { o.value } -> std::same_as<typename O::type&>;

  { O::name() } -> std::same_as<std::string>;
};

template <class W>
concept Writer =
    requires(W w, typename W::state_t::data_t t, const W::state_t& s) {
      typename W::state_t;
      requires core::concepts::State<typename W::state_t>;

      typename W::data_t;
      requires std::same_as<typename W::data_t, typename W::state_t::data_t>;

      { W::dof } -> std::convertible_to<core::index_t>;

      { w.append(t, s) } -> std::same_as<void>;
      { w.close() } -> std::same_as<void>;
    };
}  // namespace concepts
}  // namespace io