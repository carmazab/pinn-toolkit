set(LIBRARY timesteppers)

set(TEST_LIBRARY test_${LIBRARY})

set(SOURCES
  test_runge_kutta_4.cpp
  )

add_unit_test_executable(
  ${TEST_LIBRARY}
  "${SOURCES}"
  )

target_link_libraries(
  ${TEST_LIBRARY}
  PRIVATE
  ${LIBRARY}
  )