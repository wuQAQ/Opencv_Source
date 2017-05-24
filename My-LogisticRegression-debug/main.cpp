#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

float Sigmoid(float x);
void CreateSample(Mat & sample);
Mat MatSigmoid(Mat & temp);
Mat MyGradientDescent(Mat & points, float startK, float startB, float rate);

int main(void)
{
    Mat sample(100, 2, CV_32FC1);
    Mat groupX(100, 2, CV_32FC1);

    CreateSample(sample);
    //cout << sample << endl;
    MyGradientDescent(sample, 50, 50, 1);
    return 0;
}

Mat MyGradientDescent(Mat & points, float startK, float startB, float rate)
{
    Mat gx = Mat::ones(points.rows, 2, CV_32FC1);
    Mat gy = points.col(1);
    float num = points.rows;

    Mat temp = gx.col(1);
    temp = temp.mul(points.col(0));
    gx.col(1) = temp;

    Mat sgx = MatSigmoid(gx);

    Mat label(2, 1, CV_32FC1);
    Mat labelTemp = Mat::zeros(2, 1, CV_32FC1);
    label.at<float>(0, 0) = startK;
    label.at<float>(1, 0) = startB;

    float tempcost = 0.0;

    ofstream gdfile("gradientLine.txt");
    
    float x = 0.0;
    float y = 0.0;

    for (int i = 0; i < 10000; i++)
    {
        Mat minus = sgx * label - gy;
        //Mat costValue = minus.t() * minus;
        Mat mul = minus.t() * gx;
        labelTemp = label - (rate / num) * mul.t();
        
        // 计算代价函数的值
        //float cost = costValue.at<float>(0, 0) / (2 * num);
        //cout << cost << endl;
        //if (abs(tempcost - cost) < 0.2)
        //    break;
        
        // 写入文件中
        //gdfile << label.at<float>(0, 0) << " " << label.at<float>(0, 1) << " ";
        //gdfile << cost << " i" << endl;
        cout << labelTemp << endl;
        label = labelTemp;
    }
    //gdfile.close();

    //ofstream linearfile("linear.txt");
    //linearfile << x << " " << labelTemp.at<float>(1, 0) * x + labelTemp.at<float>(0, 0) << endl;
    //y = 22.0;
    //linearfile << (y-labelTemp.at<float>(0, 0))/labelTemp.at<float>(1, 0) << " " << y << endl << endl; 

    //linearfile.close();

    return label;
}

Mat MatSigmoid(Mat & temp)
{
    Mat result(temp.rows, temp.cols, CV_32FC1);

    for (int i = 0; i < temp.rows; i++)
    {
        for (int j = 0; j < temp.cols; j++)
        {
            result.at<float>(i, j) = Sigmoid(temp.at<float>(i, j));
        }
    }

    return result;
}

void CreateSample(Mat & sample)
{
    ifstream input("testSet.txt");
    ofstream sample1("sample1.txt");
    ofstream sample2("sample2.txt");
    string line;
    int count = 0;

    while (getline(input, line))
    {
        float x;
        float y;
        int label;

        istringstream record(line);
        record >> x;
        record >> y;
        record >> label;

        sample.at<float>(count, 0) = x;
        sample.at<float>(count, 1) = y;
        count++;

        if (label == 0)
            sample1 << x << " " << y << endl;
        else if (label == 1)
            sample2 << x << " " << y << endl;
    }
    sample1.close();
    sample2.close();
}

float Sigmoid(float x)
{
    return (1/(1+exp(-x)));
}