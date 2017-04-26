#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>//设置精度
#include <random>

using namespace cv;
using namespace std;

struct Tuple {
    float x;
    float y;
};

void MyKmeans(vector<Tuple> tuples, int clusters);
float getDistXY(Tuple t1, Tuple t2);
int clusterOfTuple(Tuple means[],Tuple tuple);

int main(void)
{
    const int MAX_CLUSTERS = 5;
    Mat img(500, 500, CV_8UC3);
    RNG rng(12345); //随机数产生器

    random_device rd;
    mt19937 gen(rd());
    
    for(;;)
    {
        vector<Tuple> tuples;
        Tuple tuple;
        int clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
        int sampleCount = clusterCount * rng.uniform(1, 201);

        cout << "clusterCount: " << clusterCount << endl;

        int countTemp = 0;
        for (int i = 0; i < clusterCount; i++)
        {
            Tuple center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            while (countTemp < (int)(sampleCount/clusterCount))
            {
                normal_distribution<double> normalx(center.x,img.cols*0.05);
                normal_distribution<double> normaly(center.y,img.rows*0.05);
                tuple.x = normalx(gen);
                tuple.y = normaly(gen);
                tuples.push_back(tuple);
                countTemp++;
            }
            countTemp  = 0;
        }

        img = Scalar::all(0);

        for(vector<Tuple>::size_type ix=0;ix!=tuples.size();++ix)
        {
            Point ipt = Point(tuples[ix].x, tuples[ix].y);
            circle(img, ipt, 2, Scalar(0, 0, 255), FILLED, LINE_AA);
        }

        for (int i = 0; i < clusterCount; i++)
        {
            cout << "(" << tuples[i].x << "," << tuples[i].y << ") " << endl;
        }

        randShuffle(tuples, 1, &rng);  
        imshow("clusters", img);
        MyKmeans(tuples, clusterCount);

        char key = (char)waitKey();     //无限等待
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
    return 0;
}

void MyKmeans(vector<Tuple> tuples, int clusterCount)
{
    vector<Tuple> clusters(clusterCount);
    vector<Tuple> center(clusterCount);

    //默认一开始将前K个元组的值作为k个簇的质心（均值）
    for (int i = 0; i < clusterCount; i++)
    {
        center[i].x = tuples[i].x;
        center[i].y = tuples[i].y;
    }

    cout << "center: " << endl;
    for (int i = 0; i < (int)center.size(); i++)
    {
        cout << "(" << center[i].x << "," << center[i].y << ") " << endl;
    }

    int label = 0;
    for (int i = 0; i != tuples.size(); ++i)
    {
        label = clusterOfTuple(center, tuples[i]);

    }

}

//计算两个元组间的欧几里距离
float getDistXY(Tuple t1, Tuple t2) 
{
	return sqrt((t1.x - t2.x) * (t1.x - t2.x) + (t1.y - t2.y) * (t1.y - t2.y));
}

//根据质心，决定当前元组属于哪个簇
int clusterOfTuple(Tuple means[],Tuple tuple, int clusterCount)
{
	float dist=getDistXY(means[0],tuple);
	float tmp;
	int label=0;//标示属于哪一个簇
	for(int i=1;i<clusterCount;i++)
    {
		tmp=getDistXY(means[i],tuple);
		if(tmp<dist) 
        {
            dist=tmp;
            label=i;
        }
	}
	return label;	
}