set(LIBRARY systems)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  core
)