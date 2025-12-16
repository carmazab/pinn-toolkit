set(LIBRARY core)

set(TEST_LIBRARY test_${LIBRARY})

set(SOURCES
  test_state.cpp
  )

add_unit_test_executable(
  ${TEST_LIBRARY}
  "${SOURCES}"
  )