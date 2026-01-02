#pragma once

#include <H5Epublic.h>

namespace io {
// Suppress error stack messages and diagnostics in specific contexts. See Sec.
// 9.4.4 of https://davis.lbl.gov/Manuals/HDF5-1.8.7/UG/13_ErrorHandling.html
class HDF5ErrorStackSilencer {
 public:
  HDF5ErrorStackSilencer() {
    H5Eget_auto(H5E_DEFAULT, &old_func_, &old_client_data_);
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
  }

  ~HDF5ErrorStackSilencer() {
    H5Eset_auto(H5E_DEFAULT, old_func_, old_client_data_);
  }

  HDF5ErrorStackSilencer(const HDF5ErrorStackSilencer&) = delete;
  HDF5ErrorStackSilencer& operator=(const HDF5ErrorStackSilencer&) = delete;

 private:
  H5E_auto2_t old_func_;
  void* old_client_data_;
};
}  // namespace io