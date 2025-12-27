include(CTest)
include(Catch)

function(add_unit_test_executable LIBRARY)

  set(TEST_LIBRARY test_${LIBRARY})

  cmake_parse_arguments(
    ARG
    ""
    ""
    "SOURCES;DEPENDENCIES"
    ${ARGN}
  )

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR
      "add_unit_test_executable(${LIBRARY}): no SOURCES specified"
    )
  endif()

  add_executable(${TEST_LIBRARY})

  target_sources(
    ${TEST_LIBRARY}
    PRIVATE
    ${ARG_SOURCES}
  )

  target_link_libraries(
    ${TEST_LIBRARY}
    PRIVATE
    ${LIBRARY}
    testing
    ${ARG_DEPENDENCIES}
  )

  catch_discover_tests(${TEST_LIBRARY})

endfunction()