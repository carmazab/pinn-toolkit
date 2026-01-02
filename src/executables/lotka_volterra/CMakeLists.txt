set(EXECUTABLE lotka_volterra)

add_executable(${EXECUTABLE}
  main.cpp
)

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  pinn
)