#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif

#include <math.h>

#if defined(_MSC_VER)
  #define mNaN(x) _isnan(x) || !_finite(x)
#else
  #define mNaN(x) isnan(x) || !isfinite(x)
#endif
