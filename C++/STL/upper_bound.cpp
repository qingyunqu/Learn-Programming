#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // array: int*
    int array[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int* ptr = std::upper_bound(array, array + 10, 5);
    cout << "upper_bound: " << *ptr << endl;
    ptr = std::lower_bound(array, array + 10, 5);
    cout << "lower_bound: " << *ptr << endl;

    // vector: iterator
    vector<int> arr = {1, 1, 2, 2, 3, 4, 5, 6};
    auto iter = std::upper_bound(arr.begin(), arr.end(), 2);
    cout << "upper_bound: " << *iter << endl;
    iter = std::lower_bound(arr.begin(), arr.end(), 2);
    cout << "lower_bound: " << *iter << ", " << *(++iter) << endl;

    return 0;
}