#include "NNet.h"
#include <fstream>
#include <sstream>

int main()
{
    NNet testNet;

    ifstream samplefile("sp.txt");
    sample sampleInOut[100];
    string line;
    int sinum = 0;
    int lineNum = 0;
    while(getline(samplefile, line))
    {
        istringstream record(line);
        if (lineNum % 2 == 0)
        {
            for (int j = 0; j < 25; j++)
            {
                double temp;
                record >> temp;
                sampleInOut[sinum].in.push_back(temp);
            }
        }
        else
        {
            for (int j = 0; j < 10; j++)
            {
                double temp;
                record >> temp;
                sampleInOut[sinum].out.push_back(temp);
            }
            sinum++;
        }
        lineNum++;
    }
    samplefile.close();
    vector<sample> sampleGroup(sampleInOut, sampleInOut+100);
    testNet.training(sampleGroup, 0.01);

    // // 预测测试数据，并输出结果
    vector<sample> testGroup(sampleInOut+10, sampleInOut+11);

    testNet.predict(testGroup);
    for (int i = 0; i < testGroup.size(); i++)
    {
        for (int j = 0; j < testGroup[i].in.size(); j++) cout << testGroup[i].in[j] << " ";
        cout << "-- prediction :";
        for (int j = 0; j < testGroup[i].out.size(); j++) cout << testGroup[i].out[j] << " ";
        cout << endl;
    }

    //system("pause");
    return 0;
}