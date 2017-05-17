#include <iostream>
#include <opencv2/opencv.hpp>

#include "eigenmvn.h"
#include <random>
#include <fstream>

using namespace std;
using namespace cv;

void CreatePoint();

int main(int, char** argv)
{
    Mat points(500, 2, CV_32FC1);
    int count = 0;
    string line;

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
        examplefile << dst << endl;
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

    return 0;
}


void CreatePoint()
{
  Eigen::Vector2d mean;
  Eigen::Matrix2d covar;
  mean << -1,0.5; // Set the mean
  // Create a covariance matrix
  // Much wider than it is tall
  // and rotated clockwise by a bit
  covar = genCovar(3,0.1,M_PI/5.0);

  // Create a bivariate gaussian distribution of doubles.
  // with our chosen mean and covariance
  const int dim = 2;
  Eigen::EigenMultivariateNormal<double> normX_solver(mean,covar);
  std::ofstream file_solver("samples_solver.txt");

  // Generate some samples and write them out to file
  // for plotting
  file_solver << normX_solver.samples(500).transpose() << std::endl;

  // same for Cholesky decomposition.
  covar = genCovar(3,0.1,M_PI/5.0);
  Eigen::EigenMultivariateNormal<double> normX_cholesk(mean,covar,true);
  std::ofstream file_cholesky("samples_cholesky.txt");
  file_cholesky << normX_cholesk.samples(500).transpose() << std::endl;

  file_solver.close();
  file_cholesky.close();

}