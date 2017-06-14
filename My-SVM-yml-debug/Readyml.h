#ifndef READYML_H
#define READYML_H

#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int reverseInt(int i);
Mat read_sample_yml(string fileName, int singleSample);
Mat read_label_yml(int singleSample);
Mat ChangePost(Mat & temp, vector<int> &randArray);

#endif