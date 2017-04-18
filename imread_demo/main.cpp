#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
	Mat srcImage = imread("test.jpg");
	imshow("source", srcImage);
	waitKey(0);
	return 0;
}