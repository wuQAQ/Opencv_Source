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
Mat MyGradientDescent(Mat & points, float rate);
void PointLine(Mat & weights);

int main(void)
{
    Mat sample(100, 3, CV_32FC1);
    Mat groupX(100, 3, CV_32FC1);

    CreateSample(sample);
    Mat weights = MyGradientDescent(sample, 0.001);
    PointLine(weights);
    return 0;
}

void PointLine(Mat & weights)
{
    float x1 = 0.0;
    float x2 = 0.0;

    ofstream result("result.txt");

    x1 = -4.0;
    x2 = (weights.at<float>(0, 0) + weights.at<float>(0,1) * x1)/(-weights.at<float>(0,2));
    result << x1 << " " << x2 << endl;

    x1 = 4.0;
    x2 = (weights.at<float>(0, 0) + weights.at<float>(0,1) * x1)/(-weights.at<float>(0,2));
    result << x1 << " " << x2 << endl;

    result.close();
}

Mat MyGradientDescent(Mat & points, float rate)
{
    Mat gx = Mat::zeros(points.rows, 3, CV_32FC1);
    Mat weights = Mat::ones(3, 1, CV_32FC1);
    Mat gy = points.col(2);
    
    Mat temp = Mat::ones(points.rows, 1, CV_32FC1);
    gx.col(0) += temp;
    gx.col(1) += points.col(0);
    gx.col(2) += points.col(1);
    
    
    for (int i = 0; i < 500; i++)
    {
        Mat th = gx * weights;
        Mat h = MatSigmoid(th);
        Mat error = gy - h;
        weights = weights + rate * gx.t() * error;
    }
    
    cout << weights << endl;
    return weights;
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
        sample.at<float>(count, 2) = label;
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