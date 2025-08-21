#pragma once
#if defined(_MSC_VER)
  #define likely(x)   (x)
  #define unlikely(x) (x)
#else
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#if defined(_MSC_VER)
#  define ALWAYS_INLINE __forceinline
#else
#  define ALWAYS_INLINE __attribute__((always_inline)) inline
#endif
