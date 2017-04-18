#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

Mat src;
Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void *);

int main(int argc, char **argv)
{
  /*
   * 读取图片
   */
	src = imread(argv[1], 1);

  /*
   *  将原图转换为灰度图，并进行滤波
   *  cvtColor颜色的空间转换
   *  blur均值滤波
   */
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

  /*
   * 显示图片
   */
  char* source_window = "Source";
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, src);

  /*
   * 使用Trackbar
   */
	createTrackbar("Threshold:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);

	waitKey(0);
	return 0;
}

/*
 * trackbar的回调函数
 */
void thresh_callback(int, void *)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

  /*
   * 阈值函数threshold
   * 这里采用了二值化阈值
   */
	threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY);

  /*
   * 寻找轮廓
   * void findContours(InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierar-chy,
   *                  int mode, int method, Point offset=Point())
   * InputOutputArray image：输入图像，Mat类的对象即可，且需要8位单通道图像，图像为二进制
   * OutputArrayOfArrays contours：检测到的轮廓，函数调用后的结果存储在这。每个轮廓位一个点向量
   * OutputArray hierar-chy：可选的输出向量，包含图像的拓扑信息。其作为轮廓的数量的表示
   * int mode：轮廓检索模式
   * int method：求轮廓近似的方法
   * Point offset=Point()：偏移
   */
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );

  /*
   * approxPolyDP来对指定的点集进行逼近，其逼近的精度是可设置的
   * void approxPolyDP(InputArray curve, OutputArray approxCurve, double epsilon, bool closed)；
   * InputArray curve：输入的点集 
   * OutputArray approxCurve：输出的点集，当前点集是能最小包容指定点集的。draw出来即是一个多边形； 
   * double epsilon：指定的精度，也即是原始曲线与近似曲线之间的最大距离。
   * bool closed：若为true,则说明近似曲线是闭合的，它的首位都是相连，反之，若为false，则断开。
   */
  for( int i = 0; i < contours.size(); i++ )
	{ 
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    //boundingRect来对指定的点集进行包含，使得形成一个最合适的正向矩形框把当前指定的点集都框住
    boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    minEnclosingCircle( contours_poly[i], center[i], radius[i] );
  }

  /*
   * drawContours绘制轮廓
   * void drawContours(InputOutputArray image, InputArrayOfArrays contours, int contourIdx,
   *                  const Scalar& color, int thickness=1, int lineType=8,
   *                  InputArray hierarchy=noArray(), int maxLevel=INT_MAX, Point offset=Point() )
   * InputOutputArray image：输入源图像
   * InputArrayOfArrays contours：输入轮廓，点向量
   * int contourIdx：轮廓绘制的指示变量，如果为负则绘制所有轮廓
   * const Scalar& color：轮廓的颜色
   * int thickness=1：轮廓线条的粗细
   * int lineType=8：线条的类型
   * InputArray hierarchy=noArray()：可选的层次结构
   * int maxLevel=INT_MAX：绘制轮廓的最大等级
   */
	Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
    rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
    circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
  }

  /// 显示在一个窗口
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
}