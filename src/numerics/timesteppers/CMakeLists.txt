set(LIBRARY numerics_timesteppers)

add_library(${LIBRARY}_concepts INTERFACE)

target_link_libraries(
  ${LIBRARY}_concepts
  INTERFACE
  systems_concepts
)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  ${LIBRARY}_concepts
  io_concepts
  systems_concepts
)