#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


int main()
{
	//read show
	Mat test = imread("test.jpg");
	imshow("test soucre picture", test);

	//mix
	Mat image = imread("back.jpg");
	Mat logo = imread("bUbuntu.jpg");

	imshow("back source picture", image);
	imshow("logo source", logo);

	Mat imageROI;
	imageROI=image(Rect(1,1,logo.cols,logo.rows));
	addWeighted(imageROI,0.5,logo,0.3,0.,imageROI); 

	imshow("changed", image);

	imwrite("changed.jpg", image);

	waitKey(0);
	return 0;
}