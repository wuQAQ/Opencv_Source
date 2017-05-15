#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <algorithm>

#include <fstream>
#include "sstream"
#include "vector"
#include <iostream>

using namespace cv;
using namespace std;

Point2f GetAverage(Mat & points);
void MinusAverage(Mat & points, Mat & avePoints, Point2f average);
Mat Covariance(Mat & points, Mat & avePoints);
void GetCovValue(Mat & avePoints, Mat & res);
void GetEigen(Mat & covVal, Mat & eigenvalues, Mat & eigenvectors);
void ChangeValue(Mat & avePoints, Mat & eigenvalues, Mat & eigenvectors);

int main(void)
{
    Mat points(500, 1, CV_32FC2);
    Mat avePoints(500, 2, CV_32FC1);
    Mat covVal;
    Mat eigenvalues;//特征值  
    Mat eigenvectors;//特征向量  
    int count = 0;

    string line;
    Point2f temp;

    ifstream input("samples_solver.txt");

    while (getline(input, line))
    {
        Point2f point;
        istringstream record(line);
        record >> point.x;
        record >> point.y;
        points.at<Point2f>(count) = point;
        count++;
    }

    cout << points << endl;

    cout << "points.size:" << points.rows << endl;
    
    covVal = Covariance(points, avePoints);
    GetEigen(covVal, eigenvalues, eigenvectors);
    ChangeValue(avePoints, eigenvalues, eigenvectors);
}

// 求全部点的平均值
Point2f GetAverage(Mat & points)
{
  Point2f average;
  Point2f sum(0, 0);

  Mat mc(1, 2, CV_32FC1, Scalar(0)); 

  reduce(points, mc, 0, CV_REDUCE_SUM);  

  cout << "mc:" << endl;
  cout << mc << endl;

  average.x = mc.at<float>(0,0) / points.rows;
  average.y = mc.at<float>(0,1) / points.rows;

  cout << "(" << average.x << ", " << average.y << ")" << endl;
  return average;
}

void ChangeValue(Mat & avePoints, Mat & eigenvalues, Mat & eigenvectors)
{

    float temp = -99;
    int maxLabel = -1;
    
    for (int i = 0; i < eigenvalues.rows; i++)
    {
        if (temp < eigenvalues.at<float>(i, 0))
        {
            temp = eigenvalues.at<float>(i, 0);
            maxLabel = i;
        }
    }

    Mat maxValue = eigenvectors.col(maxLabel);
    cout << "eigenvectors: " << endl << maxValue << endl;
    Mat result = avePoints * maxValue;
    cout << "eigenvectors: " << endl << maxValue << endl;
    //cout << "result: " << endl << result << endl;
    cout << "result" << endl;
    for (int i = 0; i < result.rows; i++)
    {
        cout << result.at<float>(i, 0) << endl;
    }
}

void MinusAverage(Mat & points, Mat & avePoints, Point2f average)
{
    Mat om = Mat::ones(500, 2, CV_32FC1);
    Mat temp(500, 2, CV_32FC1);
    om.col(0) *= average.x;
    om.col(1) *= average.y;

    for (int i = 0; i < om.rows; i++)
    {
        Point2f tp;
        tp = points.at<Point2f>(i);
        temp.at<float>(i, 0) = tp.x;
        temp.at<float>(i, 1) = tp.y;
    }

    // 矩阵相减
    subtract(temp, om, avePoints);

    cout << avePoints << endl;
}

Mat Covariance(Mat & points, Mat & avePoints)
{
    Point2f temp;
    Mat covRes(2, 2, CV_32FC1);

    float tempx, tempy;
    temp = GetAverage(points);

    MinusAverage(points, avePoints, temp);

    GetCovValue(avePoints, covRes);

    return covRes;
}

void GetEigen(Mat & covVal, Mat & eigenvalues, Mat & eigenvectors)
{
    cout << "eigen:" << endl;
    eigen(covVal, eigenvalues, eigenvectors);
    cout << "eigenvalues: " << endl << eigenvalues << endl;
    cout << "eigenvectors: " << endl << eigenvectors << endl;
}
// 得到协方差矩阵
void GetCovValue(Mat & avePoints, Mat & res)
{
    cout << avePoints.t() << endl;
    cout << "aa: " << endl;
    cout << avePoints.t()*(avePoints) << endl;
    res = avePoints.t()*(avePoints);
    res /= (avePoints.rows - 1);
    cout << "aa: " << endl;
    //cout << res << endl;
}