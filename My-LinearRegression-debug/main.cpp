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

void CreateSamples(Mat & points);
Mat GetCostValue(Mat & points, double midValue, double step);

int main(void)
{
    Mat points(40, 2, CV_32FC1);

    CreateSamples(points);
    Mat result = GetCostValue(points, 0, 1);

    cout << "result: " << endl << result << endl;
    return 0;
}

Mat GetCostValue(Mat & points, double midValue, double step)
{
    Mat gx = points.col(0);
    Mat gy = points.col(1);
    float num = points.rows;
    Mat resultCost(100, 1, CV_32FC1);

    cout << "num: " << num << endl;
    
    float tempStartValue = midValue - 50 * step;

    for (int i = 0; i < 100; i++)
    {
        float tempK = tempStartValue + i*step;
        Mat tempGx = tempK * gx;
        Mat tempMinus = tempGx - gy;
        Mat rT = tempMinus.t() * tempMinus;
        resultCost.at<float>(i, 0) = rT.at<float>(0, 0) / (2 * num);
    }

    return resultCost;
}

// 生成y=2x+1的样本点
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