#include <iostream>
using namespace std;

class Node {
public:
    int value;
    Node* next;
    Node() { next = nullptr; }
    Node(int i) {
        value = i;
        next = nullptr;
    }
};

class Ring {
public:
    int numNodes;
    Node* head;
    Ring() {
        numNodes = 0;
        head = nullptr;
    }
    Ring(const initializer_list<int>& I);  // initializer_list
    ~Ring();                               // Destructor
    Ring(const Ring& R);                   // Copy constructor
    Ring(Ring&& R);                        // Move constructor
    void operator=(const Ring& R);  // Copy Assignment (i.e., Lvalue assignment)
    void operator=(Ring&& R);       // Move assignment (i.e., Rvalue assignment)
    Ring ThreeTimes();  // Return a ring with the values of all nodes being
                        // three times of the value of current Ring
};

ostream& operator<<(ostream& str, const Ring& R);
void Ring::operator=(Ring&& R) {  // Move Assignment (i.e., Rvalue assignment)
    // Your code
    this->numNodes = R.numNodes;
    this->head = R.head;
    R.numNodes = 0;
    R.head = nullptr;
}
Ring::Ring(Ring&& R) {  // Move constructor
    // Your code
    this->numNodes = R.numNodes;
    this->head = R.head;
    R.numNodes = 0;
    R.head = nullptr;
}
void Ring::operator=(
        const Ring& R) {  // Copy Assignment (i.e., Lvalue assignment)
    cout << "copy assignment lvalue" << endl;
    // Your code
    this->numNodes = R.numNodes;
    if(R.head == nullptr) {
        this->head = nullptr;
    } else {
        this->head = new Node(R.head->value);
        Node* r = R.head->next;
        Node* t = this->head;
        while(r != R.head) {
            t->next = new Node(r->value);
            t = t->next;
            r = r->next;
        }
        t->next = this->head;
    }
}
Ring::Ring(const Ring& R) {  // Copy constructor
    cout << "copy constructor" << endl;
    // Your code
    this->numNodes = R.numNodes;
    if(R.head == nullptr) {
        this->head = nullptr;
    } else {
        this->head = new Node(R.head->value);
        Node* r = R.head->next;
        Node* t = this->head;
        while(r != R.head) {
            t->next = new Node(r->value);
            t = t->next;
            r = r->next;
        }
        t->next = this->head;
    }
}
Ring::~Ring() {  // Destructor
    // Your code
    Node* t = this->head;
    for(int i = 0; i < this->numNodes; i++) {
        Node* next = t->next;
        delete t;
        t = next;
    }
    this->head  = nullptr;
    this->numNodes = 0;
}
Ring::Ring(const initializer_list<int>& I) {  // initializer_list
    cout << "initializer_list constructor" << endl;
    //Your code
    this->numNodes = I.size();
    this->head = nullptr;
    int count = 1;
    Node* t = nullptr;
    for(int i : I) {
        if(count == 1) {
            this->head = new Node(i);
            t = this->head;
        } else {
            t->next = new Node(i);
            t = t->next;
        }
        count++;
    }
    if(t != nullptr) {
        t->next = this->head;
    }
}
Ring Ring::ThreeTimes() { 
    // Your code
    Ring r(*this);
    Node* t = r.head;
    if(t == nullptr)
        return r;
    do {
        t->value *= 3;
        t = t->next;
    } while(t != r.head);
    return r;
}
// ostream& operator<<(ostream& str, const Ring& R);
int main() {
    Ring R1{1, 2, 3, 4, 5};
    cout << R1 << endl;
    Ring R2{R1};
    cout << R2 << endl;
    Ring R3;
    R3 = R1;
    cout << R3 << endl;
    Ring R4;
    R4 = R1.ThreeTimes();
    cout << R4 << endl;
    return 0;
}
ostream& operator<<(ostream& str, const Ring& R) {
    // Your code
    Node* t = R.head;
    if(t == nullptr)
        return str;
    do {
        str << t->value << " ";
        t = t->next;
    } while(t != R.head);
    return str;
}