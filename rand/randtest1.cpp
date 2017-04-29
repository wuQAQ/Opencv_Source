#include <iostream>
#include <random>

using namespace std;

int main(void)
{
    default_random_engine e;
    for (size_t i = 0; i < 100; i++)
    {
        cout << e() << " ";
        if (i % 9 == 0)
            cout << endl;
    }
    cout << endl;
    return 0;
}