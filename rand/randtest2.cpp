#include <iostream>
#include <random>

using namespace std;

int main(void)
{
    uniform_int_distribution<unsigned> u(0, 9);
    default_random_engine e;
    for (size_t i = 0; i < 100; i++)
    {
        cout << u(e) << " ";
        if ((i+1) % 10 == 0)
            cout << endl;
    }
    cout << endl;
    return 0;
}