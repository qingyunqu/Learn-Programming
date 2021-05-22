#include <iostream>

using namespace std;

class Pooling {
public:
    Pooling() {
        cout << "Pooling()" << endl;
    }
    Pooling(int a){
        cout << "Pooling: " << a << endl;
    }
};

class Pooling_arm : public Pooling {
public:
    Pooling_arm(){
        cout << "Pooling_arm()" << endl;
    }
    Pooling_arm(int a) {
        cout << "Pooling_arm: " << a << endl;
    }
};

int main(){
    Pooling_arm a;
    Pooling_arm b(10);
    return 0;
}
