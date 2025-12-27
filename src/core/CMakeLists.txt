set(LIBRARY core)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  pinn_headers
)