// An example program in which the
// user can draw boxes on the screen.
//
//#include <cv.h>
//#include <highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
// Define our callback which we will install for
// mouse events.
//
void my_mouse_callback(
	int event, int x, int y, int flags, void* param
);

CvRect box;
bool drawing_box = false;
bool isRectDrawn = false;

// A litte subroutine to draw a box onto an image_copy
void draw_box( IplImage* img, CvRect rect ) {
	cvRectangle (
		img,
		cvPoint(box.x,box.y),
		cvPoint(box.x+box.width,box.y+box.height),
		cvScalar(0x00,0x00,0xff) /* blue */
	);
}

void draw_box_green( IplImage* img, CvRect rect ) {
	cvRectangle (
		img,
		cvPoint(box.x,box.y),
		cvPoint(box.x+box.width,box.y+box.height),
		cvScalar(0x00,0xff,0x00) /* green */
	);
}

int main( int argc, char* argv[] ) {

	box = cvRect(-1,-1,0,0);
	IplImage* image_input = cvLoadImage(argv[1]);
	IplImage* image = cvCloneImage( image_input );
	IplImage* image_copy = cvCloneImage( image );
	IplImage* temp = cvCloneImage( image_copy );
	cvNamedWindow( "Box Example" );
	// Here is the crucial moment that we actually install
	// the callback. Note that we set the value ‘param’ to
	// be the image_copy we are working with so that the callback
	// will have the image_copy to edit.
	//
	cvSetMouseCallback(
		"Box Example",
		my_mouse_callback,
		(void*) image_copy
	);
	// The main program loop. Here we copy the working image_copy
	// to the ‘temp’ image_copy, and if the user is drawing, then
	// put the currently contemplated box onto that temp image_copy.
	// display the temp image_copy, and wait 15ms for a keystroke,
	// then repeat…
	//
	while( 1 ) {
		//cvCopyImage( image_copy, temp );
		cvCopy( image_copy, temp );
		if( drawing_box ) draw_box( temp, box );
		cvShowImage( "Box Example", temp );
		//if( cvWaitKey( 15 )==27 ) break;
		int key = cvWaitKey( 15 );
		if(key == 27) break;
		if(isRectDrawn)
		{
			if(key == 's' || key == 'S'){
				// draw green box
				draw_box_green( image_copy, box );
				cvCopy( image_copy, image );
				
				// save roi image
				static int index = 0;
				char save_image_name[128];
				sprintf(save_image_name, "rect_%d.jpg", index++);
				cvSetImageROI(image_input, box);
				cvSaveImage(save_image_name, image_input);
				cvResetImageROI(image_input);

				isRectDrawn = false;
			}

			if(key == 'q' || key == 'Q'){
				cvCopy( image, image_copy );
				isRectDrawn = false;
			}
		}
	}
	// Be tidy
	//
	cvReleaseImage( &image_copy );
	cvReleaseImage( &temp );
	cvDestroyWindow( "Box Example" );
}

// This is our mouse callback. If the user
// presses the left button, we start a box.
// when the user releases that button, then we
// add the box to the current image_copy. When the
// mouse is dragged (with the button down) we
// resize the box.
//
void my_mouse_callback(int event, int x, int y, int flags, void* param) 
{
	IplImage* image_copy = (IplImage*) param;
	switch( event ) 
	{
		case CV_EVENT_MOUSEMOVE: {
			if( drawing_box ) {
				box.width = x-box.x;
				box.height = y-box.y;
			}
		}
		break;
		case CV_EVENT_LBUTTONDOWN: {
			drawing_box = true;
			box = cvRect(x, y, 0, 0);
		}
		break;
		case CV_EVENT_LBUTTONUP: {
			drawing_box = false;
			isRectDrawn = true;
			if(box.width<0) {
				box.x+=box.width;
				box.width *=-1;
			}
			if(box.height<0) {
				box.y+=box.height;
				box.height*=-1;
			}
			draw_box(image_copy, box);
		}
		break;
	}
}