#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int, char** argv)
{
  int dims[] = {4, 5, 6};
  Mat m3(3, dims, CV_8UC1, Scalar::all(0));

  //cout << "m3" << endl << m3.data << endl;

  Mat mm2(5, 6, CV_8UC1);
  cout << m3.cols << endl;
  for (int i = 0; i < m3.dims; i++)
  {
    cout << "m3.size:" << m3.size[i] << endl;
    for(int j = 0; j < m3.size[0]; j++)
    {
      
    }
  }

  Mat m2(4, 30, CV_8UC1, m3.data);
  cout << "m2:" << endl << m2 << endl;
  Mat m2x6 = m2.reshape(6);
  std::vector<cv::Mat> channels;
  cv::split(m2x6, channels);

  Mat p0(5, 6, CV_8UC1, m3.data + m3.step[0] * 0);
  Mat p1(5, 6, CV_8UC1, m3.data + m3.step[0] * 1);
  Mat p2(5, 6, CV_8UC1, m3.data + m3.step[0] * 2);
  Mat p3(5, 6, CV_8UC1, m3.data + m3.step[0] * 3);

  cout << "p0:" << endl << p0 << endl;
  cout << "p1:" << endl << p1 << endl;
  cout << "p2:" << endl << p2 << endl;
  cout << "p3:" << endl << p3 << endl;

  return 0;
}
