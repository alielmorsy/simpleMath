#include <iostream>
#include <tuple>
#include <variant>
#include "library/array.hpp"
#include "library/test.h"
#include "library/holders.h"
#include "library/exceptions.h"
#include "library/array_utils.h"
#include <chrono>

using namespace std::chrono;
using namespace std;

void a() {
    SMArray<int> arr = {{{1,  2, 3}, {1,  2, 3}},
                        {{34, 5, 6}, {34, 5, 6}}};
}

int main() {


//    SMArray<int> arr = {{{1,  2, 3}, {1,  2, 3}},
//                        {{34, 5, 6}, {34, 5, 6}}};
//
//    arr[ACCESS(0, 0)] = {11, 12, 13};
//    cout << static_cast<int>(arr[ACCESS(0, 0, 0)]);
    auto start = high_resolution_clock::now();
//    for (int i = 0; i < 10000; ++i) {
//
//
//        SMArray<int> arr2 = {{1, 2, 3},
//                             {1, 2, 3}};
//
//        auto a = arr + arr2;
//
//    }
    SMArray<double> arr = {{{1, 2, 3}, {1, 2, 3}},
                        {{1, 2, 3}, {1, 2, 3}}};
    SMArray<double>arr2 = {{1, 2, 3},{1, 2, 3}};

    auto a = arr / arr2;
    auto stop = high_resolution_clock::now();
    // Calculate the duration
    auto duration = duration_cast<microseconds>(stop - start);

    // Output the duration
    cout << "Execution time: " << duration.count() << " microseconds" << endl;
    for (int i = 0; i < a.shape[0]; ++i) {
        for (int j = 0; j < a.shape[1]; ++j) {
            for (int k = 0; k < a.shape[2]; ++k) {
                cout << static_cast<double>(a[ACCESS(i, j, k)]) << " ";
            }
            cout << "//";
        }
        cout << endl;
    }
    return 0;
}

