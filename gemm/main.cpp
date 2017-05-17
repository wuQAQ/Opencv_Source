#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	float va[3][1] = 
	{    
		{1},
		{0},
		{0}
	};

	Mat a(3, 1, CV_32FC1, va);
	Mat b = Mat::ones(3, 3, CV_32FC1);

	Mat c;
	gemm( a, b, 1, Mat(), 0, c, GEMM_2_T );
	cout << c;

	cout << endl << endl;
	cout << a * b << endl;
	return 0;
}