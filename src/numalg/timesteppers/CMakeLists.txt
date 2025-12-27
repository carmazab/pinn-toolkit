set(LIBRARY timesteppers)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  systems
)