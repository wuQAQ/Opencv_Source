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
int ShowPlot(const char * name, int mode);

#define PLOTSOURCE "plot 'sample1.txt' w p, 'sample2.txt' w p\n"
#define PLOTRESULT "plot 'sample1.txt' w p, 'sample2.txt' w p, 'result.txt' w l\n"

int main(void)
{
    Mat sample(100, 3, CV_32FC1);
    Mat groupX(100, 3, CV_32FC1);

    CreateSample(sample);
    ShowPlot(PLOTSOURCE, 0);
    Mat weights = MyGradientDescent(sample, 0.001);
    PointLine(weights);
    ShowPlot(PLOTRESULT, 0);
    return 0;
}

void PointLine(Mat & weights)
{
    float x1 = 0.0;
    vector<float> k;
    vector<Point2f> rp;
    
    for (int i = 0; i < weights.rows; i++)
    {
        k.push_back(weights.at<float>(i, 0));
    }

    ofstream result("result.txt");
    x1 = 4.0;
    while(x1 > -4.1)
    {
        float c = k.at(0) + k.at(1) * x1 + k.at(3) * x1;
        float a = k.at(4);
        float b = k.at(2) + k.at(5) * x1;

        float temp = b*b - 4*a*c;
        if (temp >= 0)
        {
            float r1 = (-b+sqrt(temp))/(2 * a);
            float r2 = (-b-sqrt(temp))/(2 * a);
            Point2f p;
            p.x = x1;
            p.y = r2;
            rp.push_back(p);
            result << x1 << " " << r1 << endl;
        }
        x1 -= 0.01;
    }
    for (int i = rp.size()-1; i >= 0; i--)
    {
        result << rp.at(i).x << " " << rp.at(i).y << endl;
    }
            
    result.close();
}

Mat MyGradientDescent(Mat & points, float rate)
{
    Mat gx = Mat::zeros(points.rows, 6, CV_32FC1);
    Mat weights = Mat::ones(6, 1, CV_32FC1);
    Mat gy = points.col(2);
    
    Mat temp = Mat::ones(points.rows, 1, CV_32FC1);
    Mat tempx1 = points.col(0);
    Mat tempx2 = points.col(1);
    gx.col(0) += temp;
    gx.col(1) += points.col(0);
    gx.col(2) += points.col(1);
    gx.col(3) += tempx1.mul(tempx1);
    gx.col(4) += tempx2.mul(tempx2);
    gx.col(5) += tempx1.mul(tempx2);
    
    //cout << gx << endl;
    for (int i = 0; i < 50000; i++)
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

int ShowPlot(const char * name, int mode)
{
    FILE *fp = popen("gnuplot", "w");
    if (fp ==  NULL)
        return -1;
    
    cout << name << endl;
    if (mode == 1)
    {
        fputs("set dgrid3d\n", fp);
        fputs("set contour base\n", fp);
        fputs("set cntrparam levels incremental -5,400,4000\n", fp);
        fputs("set xrange [-50:50]\n", fp);
        fputs("set yrange [-50:50]\n", fp);
        fputs("set zrange [-10:14000]\n", fp);
    }

    fputs(name, fp);
    fflush(fp);
    cin.get();
    pclose(fp);
    return 0;
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