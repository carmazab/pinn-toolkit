#pragma once

#include <cassert>

#ifndef NDEBUG
#define CORE_ASSERT(condition, message) \
  do {                                  \
    assert((condition) && (message));   \
  } while (0)
#else
#define CORE_ASSERT(condition, message) \
  do {                                  \
    (void)sizeof(condition);            \
  } while (0)
#endif