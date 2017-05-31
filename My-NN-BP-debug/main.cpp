#include "NNet.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    NNet testNet;

    vector<Mat> match_mat;

    // 学习样本
    vector<double> samplein[25];
    vector<double> sampleout[10];
    
    cout << "start" << endl;
    FileStorage readfs("sample.yml", FileStorage::READ);

    if( readfs.isOpened())
    {
        for (int i = 0; i <= 9; i++)
        {
            Mat temp;
            String mat_i = "Mat_" + to_string(i);
            readfs[mat_i] >> temp;
            match_mat.push_back(temp.clone());
        }
    }
    readfs.release();

    sample sampleInOut[20];
    for (int i = 0; i < 10; i++)
    {
        Mat temp = match_mat.at(i);
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 25; k++)
                 sampleInOut[i*2+j].in.push_back(temp.at<double>(k,j));
                 //cout << temp.at<double>(k,j);
            for (int k = 0; k < 10; k++)
            {
                if (k == i)
                    sampleInOut[i*2+j].out.push_back(1);
                else 
                    sampleInOut[i*2+j].out.push_back(0);
            }
        }
    }

    cout << "train" << endl;
    vector<sample> sampleGroup(sampleInOut, sampleInOut + 20);
    testNet.training(sampleGroup, 0.01);
    cin.get();
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