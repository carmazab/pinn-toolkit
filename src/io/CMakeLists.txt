set(LIBRARY io)

add_library(${LIBRARY}_concepts INTERFACE)

target_link_libraries(
  ${LIBRARY}_concepts
  INTERFACE
  core
)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  ${LIBRARY}_concepts
  core
  hdf5_cpp-shared
)