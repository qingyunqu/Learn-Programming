#include <iostream>

using namespace std;

struct Op {
    Op(){}
    void operator()(){
        cout << "void operator()" << endl;
    }
    static void hahaha(){
        cout << "static void hahaha" << endl;
    }
};

int main(){
    Op op;
    op();
    Op::hahaha();
}
