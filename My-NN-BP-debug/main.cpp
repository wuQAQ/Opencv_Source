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
            cout << sinum << endl;
        }
        lineNum++;
    }
    samplefile.close();

    //testNet.training(sampleGroup, 0.01);
    // 测试数据
    // vector<double> testin[4];
    // vector<double> testout[4];
    // testin[0].push_back(0.1);   testin[0].push_back(0.2);
    // testin[1].push_back(0.15);  testin[1].push_back(0.9);
    // testin[2].push_back(1.1);   testin[2].push_back(0.01);
    // testin[3].push_back(0.88);  testin[3].push_back(1.03);
    // sample testInOut[4];
    // for (int i = 0; i < 4; i++) testInOut[i].in = testin[i];
    // vector<sample> testGroup(testInOut, testInOut + 4);

    // // 预测测试数据，并输出结果
    // testNet.predict(testGroup);
    // for (int i = 0; i < testGroup.size(); i++)
    // {
    //     for (int j = 0; j < testGroup[i].in.size(); j++) cout << testGroup[i].in[j] << "\t";
    //     cout << "-- prediction :";
    //     for (int j = 0; j < testGroup[i].out.size(); j++) cout << testGroup[i].out[j] << "\t";
    //     cout << endl;
    // }

    //system("pause");
    return 0;
}