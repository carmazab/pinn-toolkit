set(LIBRARY numerics_timesteppers)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  io_concepts
  numerics_concepts
  systems_concepts
)