#include <iostream>
#include <random>
#include <vector>

using namespace std;

int main(void)
{
    default_random_engine e;
    normal_distribution<> n(4, 1.5);
    vector<unsigned> vals(9);
    for (size_t i = 0; i != 200; ++i)
    {
        unsigned v = lround(n(e));
        if (v < vals.size())
            ++vals[v];
    }

    for (size_t j = 0; j != vals.size(); ++j)
        cout << j << ": " << string(vals[j], '*') << endl;
}