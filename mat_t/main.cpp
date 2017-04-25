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
    Mat test = Mat::ones(5, 1, CV_64F);
    Mat tt = test.t();

    cout << "test:" << endl << test << endl;
    cout << "t:" << endl << tt << endl;
    
    return 0;
}
