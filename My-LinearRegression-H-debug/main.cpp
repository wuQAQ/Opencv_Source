#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

#define K 2
#define B 1
#define KNUM 100
#define BNUM 100

void CreateSamples(Mat & points);
void GetCostValue(Mat & points, float midKValue, float midBValue, float step);
double MyGradientDescent(Mat & points, float startK, float startB, float rate);

int main(void)
{
    Mat points(40, 2, CV_32FC1);

    CreateSamples(points);
    GetCostValue(points, 0, 0, 0.1);
    MyGradientDescent(points, 50, 50, 0.1);
    //cout << "result: " << endl << gd << endl;
    return 0;
}

void GetCostValue(Mat & points, float midKValue, float midBValue, float step)
{
    Mat gx = points.col(0);
    Mat gy = points.col(1);
   
    float num = points.rows;
    
    cout << "num: " << num << endl;
    
    float tempStartKValue = midKValue - (KNUM / 2) * step;
    float tempStartBValue = midBValue - (BNUM / 2) * step;

    ofstream costfile("costValue.txt");

    for (int i = 0; i < KNUM; i++)
    {
        for (int j = 0; j < BNUM; j++)
        {
            float tempK = tempStartKValue + i*step;
            float tempB = tempStartBValue + j*step;
            Mat coe = Mat::ones(40, 1, CV_32FC1);
            coe = tempB * coe;
            Mat tempGx = tempK * gx + coe;
            Mat tempMinus = tempGx - gy;
            Mat rT = tempMinus.t() * tempMinus;
            float temp =  rT.at<float>(0, 0) / (2 * num);
            //cout << temp << endl;
            costfile << i - (KNUM/2) << " " << j - (BNUM/2) << " " << temp << " i" << endl;
        }
        //cout << i << endl;
    }

    costfile.close();
}

double MyGradientDescent(Mat & points, float startK, float startB, float rate)
{
    Mat gx = Mat::ones(points.rows, 2, CV_32FC1);
    Mat gy = points.col(1);
    float num = points.rows;

    Mat temp = gx.col(1);
    temp = temp.mul(points.col(0));
    gx.col(1) = temp;

    Mat label(2, 1, CV_32FC1);
    Mat labelTemp = Mat::zeros(2, 1, CV_32FC1);
    label.at<float>(0, 0) = startK;
    label.at<float>(1, 0) = startB;

    float tempcost = 0.0;
    while(1)
    {
        Mat minus = gx * label - gy;
        Mat mul = minus.t() * gx;
        //cout << "mul: " << mul << endl;
        labelTemp = label - (rate / num) * mul.t();
        cout << "temp: " << endl << labelTemp << endl;
        //cout << "label: " << endl << label << endl;
        
        Mat costValue = minus.t() * minus;
        
        float cost = costValue.at<float>(0, 0) / (2 * num);
        cout << cost << endl;
        if (abs(tempcost - cost) < 0.15)
            break;

        label = labelTemp;
    }
    
    return 1.0;
}

// 生成y=kx+b的样本点
void CreateSamples(Mat & points)
{
    default_random_engine e;
    uniform_real_distribution<float> u(-1, 1);
    for (int i = 0; i < points.rows; i++)
    {
        float x = i / 10.0;
        float y = (K * x + B) + u(e);
        points.at<float>(i, 0) = x;
        points.at<float>(i, 1) = y;
    }
    cout << endl;

    ofstream sourcefile("source.txt");
    for (int i = 0; i < points.rows; i++)
    {
        sourcefile << points.at<float>(i, 0);
        sourcefile << " ";
        sourcefile << points.at<float>(i, 1);
        sourcefile << endl;
    }
    sourcefile.close();

    ofstream sourcelinefile("sampleline.txt");
    float tx = 0.0;
    float ty = K * tx + B;
    sourcelinefile << tx;
    sourcelinefile << " ";
    sourcelinefile << ty;
    sourcelinefile << endl;
    tx = points.rows / 10.0;
    ty = K * tx + B;
    sourcelinefile << tx;
    sourcelinefile << " ";
    sourcelinefile << ty;
    sourcelinefile << endl;

    sourcefile.close();
}