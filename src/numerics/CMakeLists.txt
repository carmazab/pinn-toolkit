set(LIBRARY numerics)

add_library(${LIBRARY}_concepts INTERFACE)

target_link_libraries(
  ${LIBRARY}_concepts
  INTERFACE
  systems_concepts
)

add_subdirectory(timesteppers)