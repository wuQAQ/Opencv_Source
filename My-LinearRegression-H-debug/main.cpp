#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

#define K 4
#define B 5
#define KNUM 100
#define BNUM 100

void CreateSamples(Mat & points);
void GetCostValue(Mat & points, float midKValue, float midBValue);
Mat MyGradientDescent(Mat & points, float startK, float startB, float rate);
int ShowPlot(const char * name, int mode);

#define PLOTSOURCE "plot 'source.txt', 'sampleline.txt' w l\n"
#define PLOTCOSTVALUE "splot 'costValue.txt' with lines\n"
#define PLOTGRADIENT "splot 'costValue.txt' with lines, 'gradientLine.txt' w l\n"
#define PLOTLINEAR "plot 'source.txt', 'linear.txt' w l\n"

int main(void)
{
    Mat points(40, 2, CV_32FC1);

    CreateSamples(points);
    ShowPlot(PLOTSOURCE, 0);

    GetCostValue(points, 0, 0);
    ShowPlot(PLOTCOSTVALUE, 1);

    Mat label = MyGradientDescent(points, 49, 49, 0.1);
    ShowPlot(PLOTGRADIENT, 0);
    cout << "K: " << label.at<float>(1, 0) << " " << "B: " << label.at<float>(0, 0) << endl;
    ShowPlot(PLOTLINEAR, 0);
    return 0;
}

void GetCostValue(Mat & points, float midKValue, float midBValue)
{
    Mat gx = points.col(0);
    Mat gy = points.col(1);
   
    float num = points.rows;
    
    float tempStartKValue = midKValue - (KNUM / 2);
    float tempStartBValue = midBValue - (BNUM / 2);

    cout << "tempStartKValue: " << tempStartKValue << endl;
    cout << "tempStartBValue: " << tempStartBValue << endl;
    
    ofstream costfile("costValue.txt");

    for (int i = 0; i < KNUM; i++)
    {
        for (int j = 0; j < BNUM; j++)
        {
            float tempK = tempStartKValue + i;
            float tempB = tempStartBValue + j;
            Mat coe = Mat::ones(40, 1, CV_32FC1);
            coe = tempB * coe;
            Mat tempGx = tempK * gx + coe;
            Mat tempMinus = tempGx - gy;
            Mat rT = tempMinus.t() * tempMinus;
            float temp =  rT.at<float>(0, 0) / (2 * num);
            costfile << tempK << " " << tempB << " " << temp << " i" << endl;
        }
    }

    costfile.close();
}

Mat MyGradientDescent(Mat & points, float startK, float startB, float rate)
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

    ofstream gdfile("gradientLine.txt");
    
    float x = 0.0;
    float y = 0.0;
    while(1)
    {
        Mat minus = gx * label - gy;
        Mat costValue = minus.t() * minus;
        
        Mat mul = minus.t() * gx;
        labelTemp = label - (rate / num) * mul.t();
        
        // 计算代价函数的值
        float cost = costValue.at<float>(0, 0) / (2 * num);
        cout << cost << endl;
        if (abs(tempcost - cost) < 0.2)
            break;
        
        // 写入文件中
        gdfile << label.at<float>(0, 0) << " " << label.at<float>(0, 1) << " ";
        gdfile << cost << " i" << endl;

        label = labelTemp;
    }
    gdfile.close();

    ofstream linearfile("linear.txt");
    linearfile << x << " " << labelTemp.at<float>(1, 0) * x + labelTemp.at<float>(0, 0) << endl;
    y = 22.0;
    linearfile << (y-labelTemp.at<float>(0, 0))/labelTemp.at<float>(1, 0) << " " << y << endl << endl; 

    linearfile.close();

    return label;
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

int ShowPlot(const char * name, int mode)
{
    FILE *fp = popen("gnuplot", "w");
    if (fp == NULL) 
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