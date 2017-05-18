#include <iostream>
#include <opencv2/opencv.hpp>

#include "eigenmvn.h"
#include <random>
#include <fstream>
#include <cstdio> 
#include <vector>

using namespace std;
using namespace cv;

#ifndef M_PI
#define M_PI REAL(3.1415926535897932384626433832795029)
#endif

void CreatePoint();

Eigen::Matrix2d genCovar(double v0,double v1,double theta);
int ShowPlot(const char * name);

int main(int, char** argv)
{
    Mat points(500, 2, CV_32FC1);
    int count = 0;
    string line;

    CreatePoint();

    ifstream input("samples_solver.txt");

    while (getline(input, line))
    {
        //Point2f point;
        float x, y;
        istringstream record(line);
        record >> x;
        record >> y;
        points.at<float>(count, 0) = x;
        points.at<float>(count, 1) = y;
        count++;
    }
    input.close();

    PCA pca_analysis(points, Mat(), CV_PCA_DATA_AS_ROW);

    cout << "mean: " << endl;
    cout << pca_analysis.mean << endl;

    cout << "eigenvalues: " << endl;
    cout << pca_analysis.eigenvalues << endl;//特征值
     
    cout << "eigenvectors: " << endl;
    cout << pca_analysis.eigenvectors << endl;//特征向量

    Mat temp = pca_analysis.eigenvectors.row(0);
    Mat dst = pca_analysis.project(points);
    cout << "new:" << endl;
    //cout << dst << endl;

    ofstream examplefile("example.txt");
    if (examplefile.is_open())
    {
        for (int i = 0; i < dst.rows; i++)
        {
            examplefile << dst.at<float>(i, 0) << " " 
                        << dst.at<float>(i, 1) << endl;
        }
        examplefile.close();
    }

    float cx = pca_analysis.mean.at<float>(0, 0);
    float cy = pca_analysis.mean.at<float>(0, 1);

    float x, y;
    ofstream centerfile("center.txt");
    if (centerfile.is_open())
    {
         x = cx + pca_analysis.eigenvectors.at<float>(0, 0);
         y = cy + pca_analysis.eigenvectors.at<float>(0, 1);
        centerfile << cx << " " << cy << endl;
        centerfile << x << " " << y << endl;

         x = cx + 0.5*pca_analysis.eigenvectors.at<float>(1, 0);
         y = cy + 0.5*pca_analysis.eigenvectors.at<float>(1, 1);
        centerfile << cx << " " << cy << endl;
        centerfile << x << " " << y << endl;
        centerfile.close();
    }

    ShowPlot("plot 'center.txt' w linespoints, 'samples_solver.txt'\n");
    ShowPlot("plot 'center.txt' w linespoints, 'samples_solver.txt','example.txt' \n");

    return 0;
}


void CreatePoint()
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
  fputs("set mouse\n", fp); 
  fputs(name, fp); 
  fflush(fp); 
  cin.get(); 
  pclose(fp); 
  return 0;
}
