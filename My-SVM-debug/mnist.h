#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int reverseInt(int i);
Mat read_mnist_image(const string fileName);
Mat read_mnist_label(const string fileName);

#endif