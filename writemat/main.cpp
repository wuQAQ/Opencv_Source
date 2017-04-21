#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int, char** argv)
{
  Mat col = Mat::ones(5, 1, CV_64F);
  Mat ccol = Mat::zeros(5, 1, CV_64F);
  Mat test = Mat::zeros(5, 10, CV_64F);

  cout << "col: " << endl << col << endl;
  cout << "ccol: " << endl << ccol << endl;
    for (int i = 0; i < 10; i++)
    {
      if (i%2 == 0)
      {
        cout << i << endl;
        test.col(i) += col.col(0);
      }
      else
      {
        test.col(i) += ccol.col(0);
      }
    }
    
    cout << "test:" << endl << test << endl;
    cout << "test size: " << test.size() << endl;
    cout << "test width: " << test.width() << endl;

    cout << "test height: " << test.height() << endl;
    return 0;
}
