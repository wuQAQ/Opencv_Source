#include <iostream>
#include <string>
#include <random>
#include <iomanip>//设置精度
#include <fstream>
using namespace std;
int main()
{
    //default_random_engine generator;//如果用这个默认的引擎，每次生成的随机序列是相同的。
    random_device rd;
    mt19937 gen(rd());

    //normal(0,1)中0为均值，1为方差
    normal_distribution<double> normal(0,1);
    ofstream ofs;
    //将结果写到文件
    string path="result.txt";
    ofs.open(path,ios::out);

    for(int n=0; n<10000; ++n) {
        for (int j = 0;j<20;++j)
        {
            ofs<<setprecision(4)<<normal(gen)<<" ";
        }   
    }
    ofs.close();

    return 0;
}