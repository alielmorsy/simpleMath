#include <iostream>
#include <complex>
#include <cassert>
#include <sm.h>



int main() {
    auto ones = sm::ones<int>(3, 3);
    ones(1, 1) = 100;
    auto repeated = ones.repeat(3, 0);
    std::cout << "repeated = " << repeated << "\n\n";
    std::cout << "=== SMArray Nested Initializer List Tests ===\n\n";

    // Test 1: 1D array
    std::cout << "Test 1: 1D Array\n";
    sm::SMArray<int> arr1d = {1, 2, 3, 4, 5};


    assert(arr1d(2) == 3);
    std::cout << "arr1d(2) = " << arr1d(2) << "\n\n";

    // Test 2: 2D array
    std::cout << "Test 2: 2D Array\n";
    sm::SMArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}};


    assert(arr2d(0, 1) == 2);
    assert(arr2d(1, 2) == 6);
    arr2d(1, 2) = 10;
    assert(arr2d(1, 2) == 10);
    std::cout << "arr2d(0,1) = " << arr2d(0, 1) << "\n";
    std::cout << "arr2d(1,2) = " << arr2d(1, 2) << "\n";
    arr2d(0,SLICE(1, -1)) = {3, 4};
    std::cout << "arr2d(0, 2) = " << arr2d(0, 2) << std::endl << std::endl;
    // Test 3: 3D array
    std::cout << "Test 3: 3D Array\n";
    sm::SMArray<int> arr3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};


    assert(arr3d(0, 0, 0) == 1);
    assert(arr3d(0, 0, 1) == 2);
    assert(arr3d(1, 1, 1) == 8);
    std::cout << "arr3d(0,0,0) = " << arr3d(0, 0, 0) << "\n";
    std::cout << "arr3d(1,1,1) = " << arr3d(1, 1, 1) << "\n\n";

    // Test 4: 4D array
    std::cout << "Test 4: 4D Array\n";
    sm::SMArray<int> arr4d = {{{{1, 2}}, {{3, 4}}}, {{{5, 6}}, {{7, 8}}}};


    assert(arr4d(0, 0, 0, 0) == 1);
    assert(arr4d(1, 1, 0, 1) == 8);
    std::cout << "arr4d(0,0,0,0) = " << arr4d(0, 0, 0, 0) << "\n";
    std::cout << "arr4d(1,1,0,1) = " << arr4d(1, 1, 0, 1) << "\n\n";

    // Test 5: Double precision
    std::cout << "Test 5: Double Array\n";
    sm::SMArray<double> arrd = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};


    assert(std::abs(arrd(1, 0) - 3.3) < 1e-10);
    std::cout << "arrd(1,0) = " << arrd(1, 0) << "\n\n";

    // Test 6: Complex numbers
    std::cout << "Test 6: Complex Array\n";
    using Complex = std::complex<double>;
    sm::SMArray<Complex> arrc = {
        {Complex(1, 2), Complex(3, 4)},
        {Complex(5, 6), Complex(7, 8)}
    };

    std::cout << "arrc(0,0) = " << arrc(0, 0) << "\n";
    std::cout << "arrc(1,1) = " << arrc(1, 1) << "\n\n";

    // Test 7: Copy and move operations
    std::cout << "Test 7: Copy and Move Operations\n";
    sm::SMArray<int> original = {{1, 2}, {3, 4}};

    // Test copy constructor


    // Test assignment

    // Test 8: Asymmetric arrays
    std::cout << "Test 8: Asymmetric Array (different inner sizes)\n";
    try {
        // Note: This implementation assumes regular arrays
        // For irregular arrays, you'd need a different approach
        sm::SMArray<int> irregular = {{1, 2, 3}, {4, 5}}; // This might not work as expected
    } catch (const std::exception &e) {
        std::cout << "Expected behavior for irregular arrays\n";
    }

    std::cout << "=== All Tests Completed Successfully! ===\n";

    return 0;
}
