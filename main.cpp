#include <iostream>
#include <tuple>
#include <variant>
#include "library/array.hpp"
#include "library/utils.h"
#include <chrono>

using namespace std::chrono;
using namespace std;

int main() {
    SMArray<int> arr = {5, 3, 99, 12, 187, 18};

    arr.sort();
//    for (int i = 0; i < arr.shape[0]; ++i) {
//        std::cout << arr[i] << endl;
//    }
    for (auto val: arr) {
        cout << val << endl;
    }
    return 0;
}

