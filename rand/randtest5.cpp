#include <iostream>
#include <random>
#include <ctime>

using namespace std;

int main(void)
{
    default_random_engine e1(2147483646);
    default_random_engine e2(time(0));

    uniform_int_distribution<unsigned> u(0, 9);
    cout << "time: " << time(0) << endl;

    cout << "e1 : " << endl;
    for (size_t i = 0; i != 100; ++i)
    {
        cout << u(e1) << " ";
        if ((i + 1) % 10 == 0)
            cout << endl;
    }

    cout << "e2 : " << endl;
    for (size_t i = 0; i != 100; ++i)
    {
        cout << u(e2) << " ";
        if ((i + 1) % 10 == 0)
            cout << endl;
    }      
}