#pragma once

#include "core/concepts.hpp"
#include "core/types.hpp"
#include "io/concepts.hpp"

namespace io {
template <class WriterT>
struct Observer {
  Observer(WriterT& writer, core::index_t write_every)
      : writer_{writer}, write_every_{write_every} {}

  template <core::StateLike StateT>
    requires WriterLike<WriterT, StateT>
  void operator()(double time, const StateT& state, core::index_t step) {
    if (step % write_every_ == 0) {
      writer_.append(time, state);
    }
  }

 private:
  WriterT& writer_;
  core::index_t write_every_;
};
}  // namespace io