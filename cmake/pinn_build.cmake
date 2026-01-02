set(LIBRARY pinn)

add_library(${LIBRARY} INTERFACE)

target_compile_features(${LIBRARY} INTERFACE cxx_std_23)

target_include_directories(
  ${LIBRARY}
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  hdf5_cpp-shared
)