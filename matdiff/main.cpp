#include "opencv2/core/core.hpp"     
#include "iostream" 
using namespace std;
using namespace cv; 

int main(void)
{
  Mat a = Mat::eye(5, 2, CV_64FC1);
  Mat b = Mat::ones(5, 1, CV_64FC1);
  //Mat temp = Mat::zeros(5, 1, CV_64FC1);
  Mat temp;
  cout << "a: " << endl << a << endl;
  cout << "b: " << endl << b << endl;

  //temp =  b.col(0) - b.col(0);
  absdiff(b, a.col(0), temp);
  cout << "temp: " << endl << temp << endl;
  temp *= 4;
  cout << "temp: " << endl << temp << endl;
  Mat t = temp.t();
  cout << "t: " << endl << t << endl;
  cout << "dot: " << endl << t * temp << endl;

  return 0;
}