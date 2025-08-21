include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Function to detect CPU features on x86-64
function(detect_x86_cpu_features target)
    if (NOT SM_ENABLE_NATIVE_OPTIMIZATION)
        message(STATUS "sm: Native optimization disabled")
        return()
    endif ()

    check_cxx_source_compiles("
        #include <immintrin.h>
        #ifdef _MSC_VER
            #include <intrin.h>
        #else
            #include <cpuid.h>
        #endif

        int main() { return 0; }
    " CAN_DETECT_CPU_FEATURES)

    if (CAN_DETECT_CPU_FEATURES)
        file(WRITE "${CMAKE_BINARY_DIR}/detect_cpu.cpp" "
            #include <iostream>
            #include <immintrin.h>
            #ifdef _MSC_VER
                #include <intrin.h>
            #else
                #include <cpuid.h>
            #endif

            bool check_avx512() {
                #ifdef _MSC_VER
                    int cpuInfo[4];
                    __cpuid(cpuInfo, 7);
                    return (cpuInfo[1] & (1 << 16)) != 0;
                #else
                    unsigned int eax, ebx, ecx, edx;
                    if (__get_cpuid_max(0, nullptr) >= 7) {
                        __cpuid_count(7, 0, eax, ebx, ecx, edx);
                        return (ebx & (1 << 16)) != 0;
                    }
                    return false;
                #endif
            }

            bool check_avx2() {
                #ifdef _MSC_VER
                    int cpuInfo[4];
                    __cpuid(cpuInfo, 7);
                    return (cpuInfo[1] & (1 << 5)) != 0;
                #else
                    unsigned int eax, ebx, ecx, edx;
                    if (__get_cpuid_max(0, nullptr) >= 7) {
                        __cpuid_count(7, 0, eax, ebx, ecx, edx);
                        return (ebx & (1 << 5)) != 0;
                    }
                    return false;
                #endif
            }

            bool check_avx() {
                #ifdef _MSC_VER
                    int cpuInfo[4];
                    __cpuid(cpuInfo, 1);
                    return (cpuInfo[2] & (1 << 28)) != 0;
                #else
                    unsigned int eax, ebx, ecx, edx;
                    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                        return (ecx & (1 << 28)) != 0;
                    }
                    return false;
                #endif
            }

            int main() {
                if (check_avx512()) std::cout << \"AVX512\" << std::endl;
                else if (check_avx2()) std::cout << \"AVX2\" << std::endl;
                else if (check_avx()) std::cout << \"AVX\" << std::endl;
                else std::cout << \"SSE2\" << std::endl;
                return 0;
            }
        ")

        try_run(CPU_DETECTION_RUN_RESULT CPU_DETECTION_COMPILE_RESULT
                "${CMAKE_BINARY_DIR}"
                "${CMAKE_BINARY_DIR}/detect_cpu.cpp"
                RUN_OUTPUT_VARIABLE CPU_FEATURES_DETECTED)

        if (CPU_DETECTION_COMPILE_RESULT AND CPU_DETECTION_RUN_RESULT EQUAL 0)
            string(STRIP "${CPU_FEATURES_DETECTED}" CPU_FEATURES_DETECTED)

            if (CPU_FEATURES_DETECTED STREQUAL "AVX512")
                if (MSVC)
                    check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
                    if (COMPILER_SUPPORTS_AVX512)
                        target_compile_options(${target} INTERFACE /arch:AVX512)
                        message(STATUS "${target}: CPU supports AVX-512, enabling AVX-512")
                    endif ()
                else ()
                    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
                    if (COMPILER_SUPPORTS_AVX512F)
                        target_compile_options(${target} INTERFACE -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl)
                        message(STATUS "${target}: CPU supports AVX-512, enabling AVX-512")
                    endif ()
                endif ()
            elseif (CPU_FEATURES_DETECTED STREQUAL "AVX2")
                if (MSVC)
                    check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
                    if (COMPILER_SUPPORTS_AVX2)
                        target_compile_options(${target} INTERFACE /arch:AVX2)
                        message(STATUS "${target}: CPU supports AVX2, enabling AVX2")
                    endif ()
                else ()
                    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
                    if (COMPILER_SUPPORTS_AVX2)
                        target_compile_options(${target} INTERFACE -mavx2 -mfma)
                        message(STATUS "${target}: CPU supports AVX2, enabling AVX2")
                    endif ()
                endif ()
            elseif (CPU_FEATURES_DETECTED STREQUAL "AVX")
                if (MSVC)
                    check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
                    if (COMPILER_SUPPORTS_AVX)
                        target_compile_options(${target} INTERFACE /arch:AVX)
                        message(STATUS "${target}: CPU supports AVX, enabling AVX")
                    endif ()
                else ()
                    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
                    if (COMPILER_SUPPORTS_AVX)
                        target_compile_options(${target} INTERFACE -mavx)
                        message(STATUS "${target}: CPU supports AVX, enabling AVX")
                    endif ()
                endif ()
            else ()
                message(STATUS "${target}: CPU supports baseline features only (SSE2)")
            endif ()
        else ()
            message(WARNING "${target}: Could not detect CPU features, falling back to compiler checks")
            fallback_to_compiler_checks(${target})
        endif ()
    else ()
        message(WARNING "${target}: CPU feature detection not supported, falling back to compiler checks")
        fallback_to_compiler_checks(${target})
    endif ()
endfunction()

# Function to detect ARM CPU features
function(detect_arm_cpu_features target)
    if (NOT SM_ENABLE_NATIVE_OPTIMIZATION)
        message(STATUS "${target}: Native optimization disabled")
        return()
    endif ()

    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            float32x4_t a = vdupq_n_f32(1.0f);
            float32x4_t b = vdupq_n_f32(2.0f);
            float32x4_t c = vaddq_f32(a, b);
            return 0;
        }
    " ARM_SUPPORTS_NEON)

    if (ARM_SUPPORTS_NEON)
        check_cxx_compiler_flag("-march=armv8-a+crypto" SUPPORTS_ARMV8_CRYPTO)
        check_cxx_compiler_flag("-march=armv8-a" SUPPORTS_ARMV8)

        if (SUPPORTS_ARMV8_CRYPTO)
            target_compile_options(${target} INTERFACE -march=armv8-a+crypto)
            message(STATUS "${target}: ARM CPU detected, enabling ARMv8-A with crypto extensions")
        elseif (SUPPORTS_ARMV8)
            target_compile_options(${target} INTERFACE -march=armv8-a)
            message(STATUS "${target}: ARM CPU detected, enabling ARMv8-A")
        else ()
            message(STATUS "${target}: ARM CPU detected, using default NEON")
        endif ()
    else ()
        message(STATUS "${target}: ARM CPU detected but NEON not available")
    endif ()
endfunction()

# Fallback function for compiler-based checks
function(fallback_to_compiler_checks target)
    if (MSVC)
        check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)

        if (COMPILER_SUPPORTS_AVX512)
            target_compile_options(${target} INTERFACE /arch:AVX512)
            message(STATUS "${target}: compiler supports AVX-512")
        elseif (COMPILER_SUPPORTS_AVX2)
            target_compile_options(${target} INTERFACE /arch:AVX2)
            message(STATUS "${target}: compiler supports AVX2")
        elseif (COMPILER_SUPPORTS_AVX)
            target_compile_options(${target} INTERFACE /arch:AVX)
            message(STATUS "${target}: compiler supports AVX")
        else ()
            message(STATUS "${target}: using baseline (SSE2)")
        endif ()
    else ()
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)

        if (COMPILER_SUPPORTS_AVX512F)
            target_compile_options(${target} INTERFACE -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl)
            message(STATUS "${target}: compiler supports AVX-512")
        elseif (COMPILER_SUPPORTS_AVX2)
            target_compile_options(${target} INTERFACE -mavx2 -mfma)
            message(STATUS "${target}: compiler supports AVX2")
        elseif (COMPILER_SUPPORTS_AVX)
            target_compile_options(${target} INTERFACE -mavx)
            message(STATUS "${target}: compiler supports AVX")
        else ()
            message(STATUS "${target}: using baseline (SSE2)")
        endif ()
    endif ()
endfunction()

check_cxx_source_compiles("int main() {
#if defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || defined(__amd64) || defined(_M_X64) || defined(_MSC_VER)
    return 0;
#else
    static_assert(false, \"Not x86-64\");
#endif
}" SM_X86)

check_cxx_source_compiles("int main() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return 0;
#else
    static_assert(false, \"Not aarch64\");
#endif
}" SM_ARM)

