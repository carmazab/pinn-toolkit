set(LIBRARY io)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  core
  hdf5_cpp-shared
)