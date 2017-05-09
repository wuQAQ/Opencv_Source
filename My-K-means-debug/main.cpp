#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

// static void help()
// {
//     cout << "\nThis program demonstrates kmeans clustering.\n"
//             "It generates an image with random points, then assigns a random number of cluster\n"
//             "centers and uses kmeans to move those cluster centers to their representitive location\n"
//             "Call\n"
//             "./kmeans\n" << endl;
// }

Mat img(500, 500, CV_8UC3);
void showImage(Mat points, int clusterCount, int sampleCount, String name);
void MyKmeans(Mat & points, int clusterCount);
int clusterOfTuple(Mat center, Point pt);
float getDistXY(Point t1, Point t2);
Point getMeans(vector<Point> pts);
vector<float> ClassifyPoint(Mat & points, int & clusterCount, Mat & center, vector<Point> & means, string name);
void DrawClassifyPoint(Mat & center,  vector<vector<Point> > & clusterGroup, vector<Point> & means,  int & clusterCount, String name);
bool UpdateCenter(Mat & center, vector<Point> & means, float accuracy);
float GetMaxDistance(Point pt1, vector<Point> pts);
bool IsCenter(vector<float> & dist);
void ChangeCenter(vector<float> & dist, Mat & center);

Scalar colorTab[] =
{
    Scalar(0, 0, 255),
    Scalar(0,255,0),
    Scalar(255,100,100),
    Scalar(255,0,255),
    Scalar(0,255,255)
};

int main( int /*argc*/, char** /*argv*/ )
{
    const int MAX_CLUSTERS = 5;
    
    RNG rng(12345);

    for(;;)
    {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
        int sampleCount = rng.uniform(1, 1001);
        Mat points(sampleCount, 1, CV_32FC2), labels;

        clusterCount = MIN(clusterCount, sampleCount);
        Mat centers;

        /* generate random sample from multigaussian distribution */
        for( k = 0; k < clusterCount; k++ )
        {
            Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                             k == clusterCount - 1 ? sampleCount :
                                             (k+1)*sampleCount/clusterCount);
            rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
        }

        
        showImage(points, clusterCount, sampleCount, "Source");
        
        randShuffle(points, 1, &rng);

        showImage(points, clusterCount, sampleCount, "randShuffle");

        MyKmeans(points, clusterCount);

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }

    return 0;
}

// 显示当前图片
void showImage(Mat points, int clusterCount, int sampleCount, String name)
{
    img = Scalar::all(0);

    for(int k = 0; k < clusterCount; k++ )
    {
        Point center;
        Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                            k == clusterCount - 1 ? sampleCount :
                                            (k+1)*sampleCount/clusterCount);
        for (int i = 0; i < pointChunk.rows; i++)
        {
            Point ipt = pointChunk.at<Point2f>(i);
            circle( img, ipt, 2, colorTab[k], FILLED, LINE_AA );
        }
    }

    imshow(name, img);
}

void MyKmeans(Mat & points, int clusterCount)
{
    vector<vector<Point> > clusterGroup(clusterCount);
    vector<float> dist;
    //随机设置中心点
    Mat center = points.rowRange(0, clusterCount);
    vector<Point> means(clusterCount);

    dist = ClassifyPoint(points, clusterCount, center, means, "first");
  
    // 重新设置中心点
    int times = 0;
    while(1)
    {
        do
        {
            String str = "times:" + to_string(times);
            dist = ClassifyPoint(points, clusterCount, center, means, str);
            times++;
            //cout << str << endl;
        } while(!UpdateCenter(center, means, 10.0));

        if (!IsCenter(dist))
        {
            ChangeCenter(dist, center);
        }
        else 
            break;
        
        cout << "center: " << center << endl;

        waitKey();
    } 
}

// 更改初始点
void ChangeCenter(vector<float> & dist, Mat & center)
{
    vector<float>::iterator max = max_element(dist.begin(), dist.end());
    vector<float>::iterator min = min_element(dist.begin(), dist.end());

    int maxLabel = distance(dist.begin(), max);
    int minLabel = distance(dist.begin(), min);

    cout << "maxLabel: " << maxLabel << endl;
    cout << "minLabel: " << minLabel << endl;

    float temp = *max - *min;
    cout << "temp: " << temp << endl;
    Point maxPt = center.at<Point2f>(maxLabel);
    Point newPt1(maxPt.x-(temp/3), maxPt.y);
    Point newPt2(maxPt.x+(temp/3), maxPt.y);
    center.at<Point2f>(maxLabel) = newPt1;
    center.at<Point2f>(minLabel) = newPt2;
}

// 检测初始点是否正确
bool IsCenter(vector<float> & dist)
{
    float distSum = 0.0;
    float average = 0.0;
    float temp = 0.0;

    for (size_t i = 0; i < dist.size(); i++)
    {
        cout << dist.at(i) << endl;
        distSum += dist.at(i);
    }

    float max = *max_element(dist.begin(), dist.end());
    float min = *min_element(dist.begin(), dist.end());
    cout << "max: " << max << endl;
    cout << "min: " << min << endl;
    temp = max - min;
    distSum -= (max + min);
    average = distSum / (dist.size() - 2);

    cout << "average/3: " << average/3 << endl;
    cout << "temp: " << temp << endl;

    return (temp <= average/3);
}

// 更新中心点
bool UpdateCenter(Mat & center, vector<Point> & means, float accuracy)
{
    float centerdist = 0.0;
    int counter = 0;

    for (int i = 0; i < center.rows; i++)
    {
        centerdist = getDistXY(center.at<Point2f>(i), means.at(i));
        center.at<Point2f>(i) = means.at(i);
        cout << centerdist << endl;
        if (centerdist < accuracy)
        {
            counter++;
        }
    }
    cout << "counter:" << counter << endl;
    return (counter==center.rows);
}

// 将点分类
vector<float> ClassifyPoint(Mat & points, int & clusterCount, Mat & center, vector<Point> & means, string name)
{
    int label = 0;
    vector<vector<Point> > clusterGroup(clusterCount);
    vector<float> dists;

    for (int i = 0; i < points.rows; i++)
    {
        label = clusterOfTuple(center, points.at<Point2f>(i));
        clusterGroup.at(label).push_back(points.at<Point2f>(i));
    }

    for (int i = 0; i < clusterCount; i++)
    {
        Point mean = getMeans(clusterGroup.at(i));
        means.at(i) = mean;
        float dist = GetMaxDistance(mean, clusterGroup.at(i));
        dists.push_back(dist);
    }

    DrawClassifyPoint(center, clusterGroup, means, clusterCount, name);
    clusterGroup.clear();

    return dists;
}

// 获取最大距离
float GetMaxDistance(Point pt1, vector<Point> pts)
{
    float dist = 0;
    float temp = 0;

    for (size_t i = 0; i < pts.size(); i++)
    {
        temp = getDistXY(pt1, pts.at(i));
        if (temp > dist)
            dist = temp;
    }
    
    return dist;
}


// 画出分类的点
void DrawClassifyPoint(Mat & center,  vector<vector<Point> > & clusterGroup, vector<Point> & means,  int & clusterCount, String name)
{
    //将已经分类的图画出来
    img = Scalar::all(0);
    for (int i = 0; i < center.rows; i++)
    {
        Point ipt = center.at<Point2f>(i);
        // int size = 2;
        // int thickness = 1;
        // Scalar color(255, 255, 255)
        circle(img, ipt, 4, Scalar(255, 255, 255), FILLED, LINE_AA);
    }

    for (int i = 0; i < clusterCount; i++)
    {
        for (size_t j = 0; j < clusterGroup.at(i).size(); j++)
        {
            Point ipt = clusterGroup.at(i).at(j);
            circle( img, ipt, 2, colorTab[i], FILLED, LINE_AA );
        }
    }

    for (size_t i = 0; i < means.size(); i++)
    {
        Point ipt = means.at(i);
        circle( img, ipt, 6, Scalar(255, 255, 255), FILLED, LINE_AA );
    }
    imshow(name, img);
}

// 计算两个元组间的欧几里距离
float getDistXY(Point t1, Point t2) 
{
	return sqrt((t1.x - t2.x) * (t1.x - t2.x) + (t1.y - t2.y) * (t1.y - t2.y));
}

// 判断该点属于哪一类
int clusterOfTuple(Mat center, Point pt)
{
    float tmp;
    float dist = getDistXY(center.at<Point2f>(0), pt);
    int label = 0;
    for (int i = 1; i < center.rows; i++)
    {
        tmp = getDistXY(center.at<Point2f>(i), pt);
        if (tmp < dist)
        {
            dist=tmp;
            label=i;
        }
    }
    return label;
}

// 计算质心
Point getMeans(vector<Point> pts)
{
    int num = pts.size();
    float meansX = 0;
    float meansY = 0;
    Point mean;

    for (int i = 0; i < num; i++)
    {
        meansX += pts[i].x;
        meansY += pts[i].y;
    }
    mean.x = meansX / num;
    mean.y = meansY / num;
    
    return mean;
}
