#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat srcImage = imread("test.jpg");
    namedWindow("SOURCE", WINDOW_AUTOSIZE);
    imshow("SOURCE", srcImage);

    Mat dstImage;
    cout << "rows: " << srcImage.rows << endl;
    cout << "cols: " << srcImage.cols << endl;
  
    cvtColor(srcImage, dstImage, COLOR_RGB2GRAY);
    namedWindow("DST", WINDOW_AUTOSIZE);
    imshow("DST", dstImage);
    waitKey(0);
    return 0;
}