#include <iostream>

using namespace std;

class Layer{
public:
	Layer(){}
	virtual void load_model(){
		cout << "Layer::load_model()" << endl;
	}
	virtual void forward(){
		cout << "Layer::forward()" << endl;
	}
};

class Pooling : public Layer {
public:
	virtual void load_model() {
    	forward();
		cout << "Pooling::load_model()" << endl;
	}
	virtual void forward() {
		cout << "Pooling::forward()" << endl;
	}
};

class Pooling_arm : public Pooling {
public:
	virtual void forward() {
		cout << "Pooling_arm::forward()" << endl;
	}
};

class Pooling_teec : public Pooling_arm {
public:
	virtual void load_model(){
    	cout << "Pooling_teec::load_model()" << endl;
	}
	virtual void forward(){
    	cout << "Pooling_teec::forward()" << endl;
	}
};

int main(){
	Layer* layer = new Pooling_arm();
	layer->forward();
	layer->load_model();
	return 0;
}
