#include "opencv2/core/core.hpp"     
#include "iostream"  

using namespace std;   
using namespace cv;  

int main(int argc,char *argv[])    
{ 
	Mat a = Mat::ones(5, 1, CV_64F);
  Mat b = Mat::eye(1, 5, CV_64F);

  Mat temp = b*a;
  
  cout << "temp:" << endl << temp << endl;
}
