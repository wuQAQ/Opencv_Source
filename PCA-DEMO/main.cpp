
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

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

    cout << "mean: " ;
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

    return 0;
}