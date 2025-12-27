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
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tests/unit>
)