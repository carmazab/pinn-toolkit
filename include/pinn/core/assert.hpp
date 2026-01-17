#pragma once

#include <cassert>

#ifndef NDEBUG
#define CORE_ASSERT(condition, message) assert((condition) && (message))
#else
#define CORE_ASSERT(condition, message) ((void)0)
#endif