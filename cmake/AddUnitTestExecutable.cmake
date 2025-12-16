include(CTest)
include(Catch)

function(add_unit_test_executable TEST_LIBRARY SOURCE_FILES)
  add_executable(
    ${TEST_LIBRARY}
    ${SOURCE_FILES}
    )

  target_link_libraries(
    ${TEST_LIBRARY}
    PRIVATE
    testing
    )

  target_include_directories(
    ${TEST_LIBRARY}
    PRIVATE
    "${CMAKE_SOURCE_DIR}/tests/unit"
    )

  catch_discover_tests(${TEST_LIBRARY})
endfunction()