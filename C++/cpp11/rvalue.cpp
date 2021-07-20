#include <iostream>

using namespace std;


class A {
public:
    int a;
    A(int a_): a(a_) {}
};

A getTemp(int val) {
    return A(val);
}

A&& getTemp1(int val) {
    return A(val); // warning: 引用了一个临时变量，右值引用本质还是引用，只不过比左值多表达了一层“将亡”的含义
}

A global(10);
A&& getTemp2() {
    return std::move(global);
}

int main() {
    A obj1 = getTemp(2);
    A&& obj2 = getTemp(3);
    A&& obj3 = getTemp1(4);
    A&& obj4 = getTemp2();
    cout << obj1.a << endl;
    cout << obj2.a << endl;
    cout << obj3.a << endl;
    cout << obj4.a << endl;
    cout << global.a << endl;
    return 0;
}
