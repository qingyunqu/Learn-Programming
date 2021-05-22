#include <iostream>

using namespace std;

class test{
public:
    virtual void load_model() = 0;
    virtual void forward() = 0;
};

class Layer : public test
{
public:
    void load_model() override {
        cout << "Layer::load_model()" << endl;
    }
    void forward() override {
        cout << "Layer::forward()" << endl;
    }
};

class Pooling : public Layer
{
public:
    void load_model() override {
        cout << "Pooling::load_model()" << endl;
    }
};

class Pooling_arm : public Pooling
{
public:
    void load_model() override {
        cout << "Pooling_arm::load_model()" << endl;
    }
};

int main(){
    Pooling *pa = new Pooling();
    Pooling_arm *ppa = static_cast<Pooling_arm*>(pa);
    ppa->load_model();
    ppa->forward();
}
