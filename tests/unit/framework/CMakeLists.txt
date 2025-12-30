set(LIBRARY testing)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Catch2::Catch2WithMain
  core
)

target_include_directories(
  ${LIBRARY}
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)