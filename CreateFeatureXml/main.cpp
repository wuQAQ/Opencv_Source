#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>

#include <string>

#include <iostream>  
#include <stdio.h>  
#include <unistd.h>  
#include <dirent.h>  
#include <stdlib.h>  
#include <sys/stat.h>   

using namespace std;
using namespace cv;

#define NUMOFIMG  100
#define thresh    100
#define max_thresh 255

string sampleDir = "samples/";
string saveDir = "resultSample/";

string savexml = "samples.yml";
string samplePrefix = "Mat_";

vector<dirent> showAllFiles (const char * dir_name);
void SaveRoiImage(string sourceName, string saveName);
void createSample();
void SaveRoiImage(string sourceName, string saveName);
Mat getEigenvalues(string path);

int main(void)
{
    createSample();
    return 0;
}

vector<dirent> showAllFiles (const char * dir_name)  
{  
    vector<dirent> filenames;
    if (NULL == dir_name)  
    {  
        cout << " dir_name is null ! " << endl; 
        exit(-1);
    }  

    struct stat s;  
    lstat(dir_name ,&s);  
    if(!S_ISDIR(s.st_mode))  
    {  
        cout << "dir_name is not a valid directory !" << endl;  
        exit(-1);
    }  
      
    struct dirent * filename;    // return value for readdir()  
    DIR * dir;                   // return value for opendir()  
    dir = opendir(dir_name);  
    if (NULL == dir)  
    {  
        cout << "Can not open dir " << dir_name << endl;  
        exit(-1);
    }  
    cout << "Successfully opened the dir !" << endl;  
      
    while ((filename = readdir(dir)) != NULL)  
    {    
        if (strcmp(filename->d_name , ".") == 0 ||   
            strcmp(filename->d_name , "..") == 0)  
            continue;  
        filenames.push_back(*filename);
    }  
    return filenames;
}   

void SaveRoiImage(string sourceName, string saveName)
{
    // 1. 读取图片
    Mat srcImage = imread(sourceName);
    //imshow("s", srcImage);

    // 2. 转换为灰度图片
    Mat srcImage_gray;
    cvtColor(srcImage, srcImage_gray, CV_BGR2GRAY);

    // 3. 转换为二值图片
    Mat binaryImage;
    threshold(srcImage_gray, binaryImage, 127, 255, THRESH_BINARY);

    // 4. 滤波
    medianBlur(binaryImage, binaryImage, 5);

    // 5. 膨胀
    Mat element = getStructuringElement(MORPH_RECT, Size(20, 20/*15, 15*/));  
    erode(binaryImage, binaryImage,element);  
    //imshow("binary", binaryImage);

    // 6. 寻找轮廓
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(binaryImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );

    for (int i = 0; i < (int)contours.size(); i++)
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        //boundingRect来对指定的点集进行包含，使得形成一个最合适的正向矩形框把当前指定的点集都框住
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }

    // 7. 根据最小轮廓把原图的字体截取下来
    Mat roi;
    srcImage(boundRect[1]).copyTo(roi);
    //imshow("roi", roi);

    // 8. 保存
    cout << saveName << endl;
    imwrite(saveName, roi);

    srcImage.release();
    srcImage_gray.release();
    binaryImage.release();
    roi.release();
}

void createSample()
{
    const char *dir = sampleDir.c_str();
    vector<dirent> dirs = showAllFiles(dir);
    FileStorage fs(savexml, FileStorage::WRITE);
    vector<Mat> numSampleEigen;

    // 1.截取局部图片
    cout << "1.截取局部图片" << endl;
    for (int i = 0; i < (int)dirs.size(); i++)
    {
        //cout << dirs.at(i).d_name << endl;
        string dirname = sampleDir + dirs.at(i).d_name + "/";
        vector<dirent> tempdir = showAllFiles(dirname.c_str());
        for (int j = 0; j < (int)tempdir.size(); j++)
        {
            string tempFileName = dirname + tempdir.at(j).d_name;
            string tempSaveName = saveDir + dirs.at(i).d_name + "-" + tempdir.at(j).d_name;
            SaveRoiImage(tempFileName, tempSaveName);
        }
    }
    cout << "end" << endl;

    // 2.获取图片特征值
    cout << "2.获取图片特征值" << endl;
    for (int i = 0; i < 10; i++)
    {
        Mat groupSample = Mat::zeros(25, NUMOFIMG/10, CV_32FC1);
        
        for (int j = 0; j < NUMOFIMG/10; j++)
        {
            string filename = saveDir + to_string(i) + "-" + to_string(j) + ".jpg";
            cout << filename << endl;
            Mat temp = getEigenvalues(filename);
            groupSample.col(j) += temp.col(0);
        }
        numSampleEigen.push_back(groupSample);
    }
    cout << "end" << endl;

    // 3.将特征值写入到yml文件下
    cout << "3.将特征值写入到yml文件下" << endl;
    for(int num = 0; num < (int)numSampleEigen.size(); num++)
    {
        string labelStr = samplePrefix + to_string(num);
        fs << labelStr << numSampleEigen.at(num);
    }
    cout << "end" << endl;
    fs.release();
}

Mat getEigenvalues(string path)
{
    Mat result(25, 1, CV_32FC1);
    Mat tempRoi;
    Mat src_gray;
    Mat src = imread(path);
    cvtColor(src, src_gray, CV_BGR2GRAY);
    int xStep = (floor)(src_gray.cols / 5);
    int yStep = (floor)(src_gray.rows / 5);

    for (int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            Rect tempRect(i*xStep, j*yStep, xStep, yStep);
            src_gray(tempRect).copyTo(tempRoi);
            //imshow("tempRoi", tempRoi);
            threshold(tempRoi, tempRoi, thresh, max_thresh, THRESH_BINARY);
            int white = countNonZero(tempRoi);
            int black = tempRoi.total()-white;
            float blackRate = (float)black / tempRoi.total();
            //float tempblack = 1.0 - blackRate;
            //将其放入25X1的Mat中
            result.at<float>(i*5+j, 0) = blackRate;
        }
    }

    return result;
}
