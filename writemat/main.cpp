#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int, char** argv)
{
    FileStorage fs("test.yml", FileStorage::WRITE);
    vector<Mat> test;
    Mat temp = Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < 9; i++)
    {
        Mat tem_frame = temp.clone();
        test.push_back(temp.clone());
        //test.at(0) = temp;
    }
    
    cout << "temp: " << endl << temp << endl;

    for(int i = 0; i < test.size(); i++)
    {
        cout << i << endl << test[i] << endl;
    }

    cout << "write" << endl;

    fs << "a" << test.at(1);
    fs << "b" << test.at(2);
    

    fs.release();

    
    return 0;
}