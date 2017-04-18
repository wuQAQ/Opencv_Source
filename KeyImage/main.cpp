#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

CvPoint pt1 = Point(0,0);
CvPoint pt2 = Point(0,0);
bool is_selecting = false;

void cvMouseCallback(int mouseEvent,int x,int y,int flags,void* param)
{
	switch(mouseEvent)
	{
	case CV_EVENT_LBUTTONDOWN:
		pt1 = Point(x,y);
		pt2 = Point(x,y);
		is_selecting = true;
		break;
	case CV_EVENT_MOUSEMOVE:
		if(is_selecting)
			pt2 = Point(x,y);
		break;
	case CV_EVENT_LBUTTONUP:
		pt2 = Point(x,y);
		is_selecting = false;
		break;
	}
	return;
}

int main(int argc,char* argv[])
{
	char* window = "img";
  Mat img = imread("test.jpg", 1);
	Mat img_show,roi; 
	img.copyTo(img_show);
	namedWindow(window,CV_WINDOW_AUTOSIZE);
	setMouseCallback(window,cvMouseCallback);
	bool shift_on = false;
	while(true)
	{
		img.copyTo(img_show);
		rectangle(img_show,pt1,pt2,Scalar(255,255,255));
		imshow(window,img_show);
		char key = cvWaitKey(10);
    char keytemp = -1;
    if (keytemp != key)
    {
      keytemp = key;
      printf("%d\n", keytemp);
    }
    
		switch(key)
		{
		//ROI平移操作
    case '\t':
			shift_on = !shift_on; break;
		case 'a':
			pt1.x--; pt2.x--; break;
		case 's':
			pt1.y++; pt2.y++; break;
		case 'd':
			pt1.x++; pt2.x++; break;
		case 'w':
			pt1.y--; pt2.y--; break;
		
               //ROI放大和缩小，主要是对初始设置的ROI区域的边缘进行平移操作
		case '1':
			if(shift_on) pt1.x--; 
			else pt2.x--;
			break;
		case '2':
			if(shift_on) pt2.y++; 
			else pt1.y++;
			break;
		case '3':
			if(shift_on) pt2.x++;
			else pt1.x++;
			break;
		case '4':
			if(shift_on) pt1.y--;
			else pt2.y--;
			break;

		//回车确定最终ROI区域的截取，并将其保存下来
		case '\n':
			roi=img(Rect(pt1.x,pt1.y,std::abs(pt2.x-pt1.x),std::abs(pt2.y-pt1.y)));
			imwrite("ROI.jpg",roi);
      cout << "write ok" << endl;
			break;
		};

		if(key==27) break;
	}
	return 0;
}