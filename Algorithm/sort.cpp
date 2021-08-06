#include <iostream>
#include <vector>

using namespace std;

void quick_sort(vector<int>& nums, int l, int r) {
    if(l >= r) return;
    int first = l, last = r, key = nums[l];
    while(first < last) {
        while(first < last && nums[last] >= key) {
            last--;
        }
        nums[first] = nums[last];
        while(first < last && nums[first] <= key) {
            first++;
        }
        nums[last] = nums[first];
    }
    nums[first] = key;
    quick_sort(nums, l, first - 1);
    quick_sort(nums, first + 1, r);
}

int main() {
    vector<int> nums = {2, 1, 2, 3, 4, 5, 1};
    quick_sort(nums, 0, nums.size() - 1);
    for(auto i : nums) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
