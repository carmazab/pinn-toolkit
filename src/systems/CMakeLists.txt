set(LIBRARY systems)

add_library(${LIBRARY}_concepts INTERFACE)

target_link_libraries(
  ${LIBRARY}_concepts
  INTERFACE
  core
)

add_subdirectory(ode)