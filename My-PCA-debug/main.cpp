#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <algorithm>

#include "eigenmvn.h"
#include <fstream>
#include "sstream"
#include "vector"
#include <iostream>

using namespace cv;
using namespace std;

#ifndef M_PI
#define M_PI REAL(3.1415926535897932384626433832795029)
#endif

Point2f center;
Point2f GetAverage(Mat & points);
void MinusAverage(Mat & points, Mat & avePoints, Point2f average);
Mat Covariance(Mat & points, Mat & avePoints);
void GetCovValue(Mat & avePoints, Mat & res);
void GetEigen(Mat & covVal, Mat & eigenvalues, Mat & eigenvectors);
void ChangeValue(Mat & avePoints, Mat & eigenvalues, Mat & eigenvectors);

int ShowPlot(const char * name);
Eigen::Matrix2d genCovar(double v0,double v1,double theta);
void CreatePoint(void);

Mat tempPoints;
int main(void)
{
    Mat points(500, 2, CV_32FC1);
    Mat avePoints(500, 2, CV_32FC1);
    Mat covVal;
    Mat eigenvalues;//特征值  
    Mat eigenvectors;//特征向量  
    
    string line;
    Point2f temp;

    // 生成随机点
    CreatePoint();

    // 读取随机点
    int count = 0;
    ifstream input("samples_solver.txt");
    while (getline(input, line))
    {
        float tmpx;
        float tmpy;
        istringstream record(line);
        record >> tmpx;
        record >> tmpy;
        points.at<float>(count, 0) = tmpx;
        points.at<float>(count, 1) = tmpy;
        count++;
    }
    tempPoints = points;
    
    // 计算协方差矩阵
    covVal = Covariance(points, avePoints);
    
    // 获取特征值与特征向量
    GetEigen(covVal, eigenvalues, eigenvectors);
    cout << "eigenvalues: " << endl;
    cout << eigenvalues << endl;
    cout << "eigenvectors: " << endl;
    cout << eigenvectors << endl;

    // 降低维度
    ChangeValue(avePoints, eigenvalues, eigenvectors);

    ShowPlot("plot 'example.txt' w linespoints, 'samples_solver.txt','result.txt' \n");
}

// 求全部点的平均值
Point2f GetAverage(Mat & points)
{
  Point2f mean;

  Mat mc(1, 2, CV_32FC1, Scalar(0)); 

  reduce(points, mc, 0, CV_REDUCE_SUM);  

  mean.x = mc.at<float>(0,0) / points.rows;
  mean.y = mc.at<float>(0,1) / points.rows;

  return mean;
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

    // cout << "maxValue: " << endl;
    // cout << maxValue << endl;
    Mat result;
    gemm(avePoints, maxValue.t(), 1, Mat(), 0, result, GEMM_2_T );

    float k = 0.0;
    float a = 0.0;
    ofstream examplefile("example.txt");
    float cx = center.x;
    float cy = center.y;
    if (examplefile.is_open())
    {
        float x, y;
        x = cx + eigenvectors.at<float>(0, 0);
        y = cy + eigenvectors.at<float>(0, 1);
        examplefile << cx << " " << cy << endl;
        examplefile << x << " " << y << endl;

        a = atan2(y-cy, x-cx);
        
        x = cx + 0.5*eigenvectors.at<float>(1, 0);
        y = cy + 0.5*eigenvectors.at<float>(1, 1);
        examplefile << cx << " " << cy << endl;
        examplefile << x << " " << y << endl;

        examplefile.close();
    }

    Mat resPoints(500, 2, CV_32FC1);
    
    for (int i = 0; i < result.rows; i++)
    {
        resPoints.at<float>(i, 0) = cx+result.at<float>(i, 0) * cos(a);
        resPoints.at<float>(i, 1) = cy + result.at<float>(i, 0) * sin(a);
    }

    ofstream resultfile("result.txt");
    if (resultfile.is_open())
    {
        for (int i = 0; i < resPoints.rows; i++)
        {
            resultfile << resPoints.at<float>(i, 0);
            resultfile << " ";
            resultfile << resPoints.at<float>(i, 1);
            resultfile << endl;
        }
    }


}

// 减去平均值
void MinusAverage(Mat & points, Mat & avePoints, Point2f average)
{
    Mat om = Mat::ones(500, 2, CV_32FC1);
    om.col(0) *= average.x;
    om.col(1) *= average.y;

    // 矩阵相减
    subtract(points, om, avePoints);
}

Mat Covariance(Mat & points, Mat & avePoints)
{
    Point2f temp;
    Mat covRes(2, 2, CV_32FC1);

    // 计算平均值
    temp = GetAverage(points);
    cout << "mean: " << endl;
    cout << temp << endl;
    center = temp;
    // 减去平均值
    MinusAverage(points, avePoints, temp);

    // 获取协方差
    GetCovValue(avePoints, covRes);

    return covRes;
}

// 获取特征值与特征向量
void GetEigen(Mat & covVal, Mat & eigenvalues, Mat & eigenvectors)
{
    eigen(covVal, eigenvalues, eigenvectors);
}

// 得到协方差矩阵
void GetCovValue(Mat & avePoints, Mat & res)
{
    res = avePoints.t()*(avePoints);
    res /= (avePoints.rows - 1);
}

// 生成随机点
void CreatePoint(void)
{
    Eigen::Vector2d mean;
    Eigen::Matrix2d covar;

    mean << -1,0.5; // Set the mean

    covar = genCovar(3,0.1,M_PI/5.0);

    Eigen::EigenMultivariateNormal<double> normX_solver(mean,covar);
    std::ofstream file_solver("samples_solver.txt");

    file_solver << normX_solver.samples(500).transpose() << std::endl;

    file_solver.close();
}

Eigen::Matrix2d genCovar(double v0,double v1,double theta)
{
  Eigen::Matrix2d rot = Eigen::Rotation2Dd(theta).matrix();
  return rot*Eigen::DiagonalMatrix<double,2,2>(v0,v1)*rot.transpose();
}

int ShowPlot(const char * name)
{
  FILE *fp = popen("gnuplot", "w");
  if (fp == NULL) 
    return -1; 

  cout << name << endl;
  fputs(name, fp); 
  fflush(fp); 
  cin.get(); 
  pclose(fp); 
  return 0;
}
