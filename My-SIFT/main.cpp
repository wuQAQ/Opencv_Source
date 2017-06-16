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
#include <cmath>

typedef struct ImageSt {
    float levelSigma;
    int levelSigmaLength;
    float absoluteSigma;
    Mat *level;
} imageLevels;

typedef struct ImageSt1 {
    int row,col;
}

#define MYDEBUG
void MyDebug()
{
    #ifdef MYDEBUG 
    waitKey(0);
    #endif
}

int main()
{
    // 声明
    Mat src;
    Mat src_gray;
    Mat DoubleSizeImage;

    Mat mosaic1;
    Mat mosaic2;
    
    Mat mosaicHorizen1;
    Mat mosaicHorizen2;
    Mat mosaicVertical1;

    Mat image1Mat;
    Mat tempMat;

    // 1.读取图片
    src = imread("demo1.jpg");
    MyDebug();
    
    // 2.转为灰度图
    cvtColor(src, src_gray, CV_BGR2GRAY);
    


    // 1.SIFT算法: 图像预处理

    // 2.SIFT算法：建立高斯金字塔

    // 3.SIFT算法: 特征位置检测，确定特征点位置

    // 4.SIFT算法: 计算高斯图像的梯度方向和幅值，计算各个特征点主方向

    // 5.SIFT算法: 抽取各个特征点处的特征描述字

}