set(SOURCE
        SMArray.cpp

)

add_library(sm ${SOURCE})

target_include_directories(sm PUBLIC "../include")