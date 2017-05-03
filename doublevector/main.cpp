#include <iostream>
#include <vector>

using namespace std;
int main(void)
{
    int n = 10;
    vector<vector<int> > test(n);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            test.at(i).push_back(j);
        }
    }

    for (int i = 0; i < test.size(); i++)
    {
        for (int j = 0; j < test.at(i).size(); j++)
            cout << test.at(i).at(j) << " ";
        cout << endl;
    }
    return 0;
}