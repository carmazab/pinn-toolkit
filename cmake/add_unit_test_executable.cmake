function(add_unit_test_executable TEST_LIBRARY)

  cmake_parse_arguments(
    ARG
    ""
    ""
    "SOURCES;LINK_LIBS"
    ${ARGN}
  )

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR
      "add_unit_test_executable(${LIBRARY}): no SOURCES specified"
    )
  endif()

  add_executable(${TEST_LIBRARY})

  target_compile_features(${TEST_LIBRARY} PRIVATE cxx_std_23)

  target_sources(
    ${TEST_LIBRARY}
    PRIVATE
    ${ARG_SOURCES}
  )

  target_link_libraries(
    ${TEST_LIBRARY}
    PRIVATE
    testing
    ${ARG_LINK_LIBS}
  )

  catch_discover_tests(${TEST_LIBRARY})

endfunction()