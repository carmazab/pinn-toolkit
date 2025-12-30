add_library(pinn_headers INTERFACE)

target_include_directories(
  pinn_headers
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

add_subdirectory(core)
add_subdirectory(executables)
add_subdirectory(io)
add_subdirectory(numerics)
add_subdirectory(systems)