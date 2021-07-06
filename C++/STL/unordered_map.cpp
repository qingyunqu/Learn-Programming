#include <unordered_map>
#include <iostream>
#include <map>

int main() {
    std::unordered_map<int, int> map;
    map[1]++;
    map[2] += 1;
    std::cout << map[4] << std::endl;

    std::map<int, int> map1;
    std::cout << map1[1] << std::endl;
    std::cout << map1.at(2) << std::endl;
    return 0;
}