#include <iostream>
#include <random>
#include <vector>

using namespace std;

int main(void)
{
    default_random_engine e1;
    default_random_engine e2(2147483646);
    
    default_random_engine e3;
    e3.seed(32767);
    default_random_engine e4(32767);
    for (size_t i = 0; i != 100; ++i)
    {
        if (e1() == e2())
            cout << "unseeded match at iteration: " << i << endl;
        if (e3() != e4())
            cout << "seeded differs at iteration: " << i << endl;
    }    
}