#include <iostream>
#include <random>
#include <vector>

using namespace std;

vector<unsigned> bad_randVec();
vector<unsigned> good_randVec();
int main(void)
{
    vector<unsigned> v1(bad_randVec());
    vector<unsigned> v2(bad_randVec());

    cout << ((v1==v2) ? "equal" : "not equal") << endl;

    vector<unsigned> v3(good_randVec());
    vector<unsigned> v4(good_randVec());

    cout << ((v3==v4) ? "equal" : "not equal") << endl;
}

vector<unsigned> bad_randVec()
{
    default_random_engine e;
    uniform_int_distribution<unsigned> u(0,9);
    vector<unsigned> ret;
    for (size_t i = 0; i < 100; ++i)
        ret.push_back(u(e));
    return ret;
}

vector<unsigned> good_randVec()
{
    static default_random_engine e;
    static uniform_int_distribution<unsigned> u(0, 9);
    vector<unsigned> ret;
    for (size_t i = 0; i < 100; ++i)
        ret.push_back(u(e));
    return ret;
}