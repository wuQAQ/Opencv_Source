#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <ctime>

using namespace std;
using namespace cv;

Mat img(500, 500, CV_8UC3);

struct Tuple {
    float x;
    float y;
};

int main(void)
{
    vector<Tuple> tuples;
    Tuple tuple;

    while(1)
    {
    const int MAX_CLUSTERS = 5;
    default_random_engine randx(12345);
    default_random_engine randy(54321);
    default_random_engine randc(time(0));

    uniform_int_distribution<unsigned> u(0, 500);
    uniform_int_distribution<unsigned> ce(2, MAX_CLUSTERS);

    int clusterCount = ce(randc);
    cout << time(0) << endl;
    cout << clusterCount << endl;
    }
    return 0;
}