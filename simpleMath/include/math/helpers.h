#pragma once

#include <macros.h>

#define MAX_NDIM 5




#define CALCULATE_OFFSET_STEP for (int d = 0; d < ndim; d++) { \
            offset_step_a[d] = broadcast_a[d] ? 0 : stride_a[d];\
            offset_step_b[d] = broadcast_b[d] ? 0 : stride_b[d];\
            }
