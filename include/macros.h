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

#define CHUNK_SIZE 1024

// cross-compiler loop unrolling
#if defined(_MSC_VER)      // MSVC
    #define PRAGMA_UNROLL(n) __pragma(loop(unroll, n))
#elif defined(__clang__)   // Clang
    #define PRAGMA_UNROLL(n) _Pragma("clang loop unroll_count(n)")
#elif defined(__GNUC__)    // GCC
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define PRAGMA_UNROLL(n) _Pragma(STR(GCC unroll n))
#else
    #define PRAGMA_UNROLL(n)  // fallback: nothing
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define USE_SVML
#endif
