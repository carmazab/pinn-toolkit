set(LIBRARY testing)

add_library(${LIBRARY} INTERFACE)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Catch2::Catch2WithMain
  )

target_include_directories(
  ${LIBRARY}
  INTERFACE
  "${CMAKE_SOURCE_DIR}/tests/unit"
  )