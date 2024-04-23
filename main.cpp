#include <iostream>
#include <tuple>
#include <variant>
#include "library/array.hpp"
#include "library/test.h"
#include "library/holders.h"
#include "library/exceptions.h"
#include <chrono>

using namespace std::chrono;
using namespace std;

void a() {
    SMArray<int> arr = {{{1,  2, 3}, {1,  2, 3}},
                        {{34, 5, 6}, {34, 5, 6}}};
}

int main() {


    SMArray<int> arr = {{{1,  2, 3}, {1,  2, 3}},
                        {{34, 5, 6}, {34, 5, 6}}};

    arr[ACCESS(0, 0)] = {11, 12, 13};
    cout << static_cast<int>(arr[ACCESS(0, 0, 0)]);
    return 0;
}

