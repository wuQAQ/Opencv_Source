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
int clusterOfTuple(vector<Tuple> means,Tuple tuple, int clusterCount);
float getVal(vector<vector<Tuple> > clusters, vector<Tuple> means, int clusterCount);
Tuple getMeans(vector<Tuple> cluster);

Mat img(500, 500, CV_8UC3);

Scalar colorTab[] =     //因为最多只有5类，所以最多也就给5个颜色
{
    Scalar(0, 0, 255),
    Scalar(0,255,0),
    Scalar(255,100,100),
    Scalar(255,0,255),
    Scalar(0,255,255)
};


int main(void)
{
    const int MAX_CLUSTERS = 5;
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
    vector<vector<Tuple> > clusters(clusterCount);
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
    for (int i = 0; i != (int)tuples.size(); ++i)
    {   
        label = clusterOfTuple(center, tuples[i], clusterCount);
        clusters[label].push_back(tuples[i]);
    }

    img = Scalar::all(0);

    for (label = 0; label < clusterCount; label++)
    {
        cout << "第" << label+1 << "个簇: " << endl;
        vector<Tuple> t = clusters[label];
        for (int i = 0; i < (int)t.size(); i++)
        {
            cout << "(" << t[i].x << "," << t[i].y << ")" << "  ";
            Point ipt = Point(t[i].x, t[i].y);
            circle(img, ipt, 2, colorTab[label], FILLED, LINE_AA);
        }
        cout << endl;
    }

    imshow("kmeans", img);

    float oldVal = -1;
    float newVal = getVal(clusters, center, clusterCount);

    int name = 0;
    while(abs(newVal - oldVal) >= 1)
    {
        for (int i = 0; i < clusterCount; i++)
        {
            center[i] = getMeans(clusters[i]);
        }

        oldVal = newVal;
        newVal = getVal(clusters, center, clusterCount);
        for (int i = 0; i < clusterCount; i++)
        {
            clusters[i].clear();
        }

        for (int i = 0; i != (int)tuples.size(); ++i)
        {
            label = clusterOfTuple(center, tuples[i], clusterCount);
            clusters[label].push_back(tuples[i]);
        }

        img = Scalar::all(0);
        for (label = 0; label < clusterCount; label++)
        {
            cout << "第" << label+1 << "个簇: " << endl;
            vector<Tuple> t = clusters[label];
            for (int i = 0; i < (int)t.size(); i++)
            {
                cout << "(" << t[i].x << "," << t[i].y << ")" << "  ";
                Point ipt = Point(t[i].x, t[i].y);
                circle(img, ipt, 2, colorTab[label], FILLED, LINE_AA);
            }
            cout << endl;
        }
        String str = "times:" + to_string(name);
        imshow(str, img);
        cout << "newVal: " << newVal << endl;
        cout << "oldVal: " << oldVal << endl;
        cout << "value: " << abs(newVal - oldVal) << endl;
    }


}

//计算两个元组间的欧几里距离
float getDistXY(Tuple t1, Tuple t2) 
{
	return sqrt((t1.x - t2.x) * (t1.x - t2.x) + (t1.y - t2.y) * (t1.y - t2.y));
}

//根据质心，决定当前元组属于哪个簇
int clusterOfTuple(vector<Tuple> means,Tuple tuple, int clusterCount)
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

//获得给定簇集的平方误差
float getVal(vector<vector<Tuple> > clusters, vector<Tuple> means, int clusterCount)
{
	float var = 0;
	for (int i = 0; i < clusterCount; i++)
	{
		vector<Tuple> t = clusters[i];
		for (int j = 0; j < (int)t.size(); j++)
		{
			var += getDistXY(t[j],means[i]);
		}
	}
	//cout<<"sum:"<<sum<<endl;
	return var;
}

//获得当前簇的均值（质心）
Tuple getMeans(vector<Tuple> cluster)
{
	int num = cluster.size();
	double meansX = 0, meansY = 0;
	Tuple t;
	for (int i = 0; i < num; i++)
	{
		meansX += cluster[i].x;
		meansY += cluster[i].y;
	}
	t.x = meansX / num;
	t.y = meansY / num;
	return t;
	//cout<<"sum:"<<sum<<endl;
}