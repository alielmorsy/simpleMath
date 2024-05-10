#include <iostream>
#include <tuple>
#include <variant>
#include "library/array.hpp"
#include "library/utils.h"
#include <chrono>

using namespace std::chrono;
using namespace std;

void a() {
    SMArray<int> arr = {{{1,  2, 3}, {1,  2, 3}},
                        {{34, 5, 6}, {34, 5, 6}}};
}

//8626
int main() {
    auto start = high_resolution_clock::now();

    SMArray<double> arr = {
            {{1, 2, 3}, {1, 2, 3}},
            {{1, 2, 3}, {1, 2, 3}}
    };
    auto arr2 = ones<double>(arr.shape, arr.ndim);

    auto a = arr.transpose();
    auto stop = high_resolution_clock::now();
    // Calculate the duration
    auto duration = duration_cast<microseconds>(stop - start);

    // Output the duration
    for (int i = 0; i < a.shape[0]; ++i) {
        for (int j = 0; j < a.shape[1]; ++j) {
            for (int k = 0; k < a.shape[2]; ++k) {
                auto access = ACCESS(i, j, k);
                cout << static_cast<double>(a[access]) << " ";
            }
            cout << "//";
        }
        cout << endl;
    }
    cout << "Execution time: " << duration.count() << " microseconds" << endl;
    delete arr2;
    return 0;
}

