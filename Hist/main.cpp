#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    Mat Image=imread("test.jpg");
    //Mat Image_gray(Image.size(), 1);
    cvtColor(Image,Image,CV_BGR2GRAY);

    imshow("a", Image);
    const int channels[1]={0};
    const int histSize[1]={2};
    float hranges[2]={0,255};
    const float* ranges[1]={hranges};
    MatND hist;
    threshold(Image, Image, 127, 255, THRESH_BINARY);
    calcHist(&Image,1,channels,Mat(),hist,1,histSize,ranges);
    cout << "Hist:" << endl << hist << endl;
    int num = countNonZero(Image);
    //imshow("b", hist);
    cout << "num: " << num << endl;
    waitKey(0);
    return 0;
}