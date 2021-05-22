#include <iostream>

using namespace std;

class Layer {
public:
    virtual void forward() = 0;
    virtual void load() = 0;
};

class Pooling_arm : public Layer {
public:
    virtual void forward() {
        cout << "Pooling_arm::forward()" << endl;
    }
    virtual void load() {
        cout << "Pooling_arm::load()" << endl;
    }
};

class Pooling_cuda : public Layer {
public:
    virtual void forward() {
        cout << "Pooling_cuda::forward()" << endl;
    }
};

int main(){
    Layer* layer1 = new Pooling_arm();
    layer1->forward();
    Layer* layer2 = new Pooling_cuda();
    layer2->forward();
    return 0;
}
