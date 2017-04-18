/*
 * createTrackbar创建滑动条，往往和回调函数配合使用
 */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

#define WINDOW_NAME "TEST"

const int g_nMaxAlphaValue = 100;
int g_nAlphaValueSlider;
double g_dAlphaValue;
double g_dBetaValue;

/*
 * 声明存储变量
 */
Mat g_srcImage1;
Mat g_srcImage2;
Mat g_dstImage;

/*
 * createTrackbar的回调函数
 */
void on_Trackbar(int , void*)
{
	g_dAlphaValue = (double) g_nAlphaValueSlider/g_nMaxAlphaValue;
	g_dBetaValue = (1.0 - g_dAlphaValue);

	addWeighted(g_srcImage1, g_dAlphaValue, g_srcImage2, g_dBetaValue, 0.0, g_dstImage);
	imshow(WINDOW_NAME, g_dstImage);
}

int main(int argc, char **argv)
{
	/*
	 * 读取两张图片
	 */
	g_srcImage1 = imread("1.jpg");
	g_srcImage2 = imread("2.jpg");
	if(!g_srcImage1.data) {
		printf("source1 error \n");
	}
	if(!g_srcImage2.data) {
		printf("source2 error \n");
	}

	/*
	 * 设置滑动条的初始值
	 */
	g_nAlphaValueSlider = 70;

	namedWindow(WINDOW_NAME, 1);
	char TrackbarName[50];
	sprintf(TrackbarName, "透明值 %d", g_nMaxAlphaValue);

  /*
	 * 创建滑动条
	 * 参数：1、TrackbarName 滑动条的名字
	 *			2、WINDOW_NAME 窗口的名字，滑动条依附于什么窗口
	 *			3、int* 型的value值，表示滑动条的初始位置
	 *			4、int 型的count，表示滑块可以达到的最大位置的值
	 *			5、TrabarCallback 类型的 onChange，指向回调函数的指针
	 *   		6、void* 型的userdata，用户传给回调函数的数据
	 */
	createTrackbar(TrackbarName, WINDOW_NAME, &g_nAlphaValueSlider, g_nMaxAlphaValue, on_Trackbar);

  /*
	 * 结果在回调函数中显示
	 */
	on_Trackbar(g_nAlphaValueSlider, 0);

	waitKey(0);

	return 0;
}