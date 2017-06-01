#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <stdlib.h>

using namespace std;
using namespace cv;

int main(int, char** argv)
{
    FileStorage fs("/home/wuqaq/DISK/project/QT/drawimage/test.yml", FileStorage::READ);
    vector<Mat> test;
    Mat temp(25, 10, CV_64F);

    if (fs.isOpened())
    {
      fs["Mat_0"] >> temp;
      cout << "Mat_0:" << endl << temp << endl;
    }
    else 
    {
      cout << "close" << endl;
    }
   
    fs.release();

    
    return 0;
}
