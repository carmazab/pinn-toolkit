set(LIBRARY core)

add_library(${LIBRARY} INTERFACE)

target_include_directories(
  ${LIBRARY}
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)