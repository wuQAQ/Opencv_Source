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
    testNet.training(sampleGroup, 0.05);

    testNet.predict(sampleGroup);
    
    return 0;
}