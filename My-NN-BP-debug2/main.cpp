#include "NNet.h"
#include <fstream>
#include <sstream>
#include <ctime>

int main()
{
    BpNet testNet;

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
    clock_t start_time=clock();
    testNet.training(sampleGroup, 0.1);
    clock_t end_time=clock();
    cout<< "Running time is: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC<<"s"<<endl;//输出运行时间

    // // 预测测试数据，并输出结果
    vector<sample> testGroup(sampleInOut+10, sampleInOut+11);
    for (int i = 0; i < testGroup.size(); i++)
    {
        for (int j = 0; j < testGroup[i].out.size(); j++) cout << testGroup[i].out[j] << endl;
    }

    testNet.predict(testGroup);
    for (int i = 0; i < testGroup.size(); i++)
    {
        cout << "-- prediction :";
        for (int j = 0; j < testGroup[i].out.size(); j++) cout << testGroup[i].out[j] << " ";
        cout << endl;
    }

    //system("pause");
    return 0;
}