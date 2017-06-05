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

int GetBigOrLitterEndian(void);
vector<dirent> showAllFiles (const char * dir_name);
void SaveRoiImage(string sourceName, string saveName);

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

int main(void)
{
    string sampleDir = "samples/";
    string saveDir = "resultSample/";
    const char *dir = sampleDir.c_str();
    vector<dirent> dirs = showAllFiles(dir);

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
    

    // Mat threshold_output;
    // resize(roi_gray, threshold_output, Size(10, 10), 0, 0, 3); 

    //threshold_output = ~threshold_output/255;
    //cout << threshold_output.channels() << endl;
    // for (int i = 0; i < threshold_output.rows; i++)
    // {
    //     for (int j = 0; j < threshold_output.cols; j++)
    //     {
    //         cout << (int)threshold_output.at<uchar>(i, j);
    //     }
    //     cout << endl;
    // }

    // 7.判断主机字节序
    int byteorder = GetBigOrLitterEndian();
    if (0 == byteorder)
    {
        cout << "此程序不适合大端字节数的主机" << endl;
        return -1;
    }
    
    // 8.制作二进制文件
    // ofstream infile("samples.idx3-ubyte", ios::binary);
    // if (infile.is_open())
    // {
    //     int magicNumber = 0x0803;
    //     cout << hex << temp << endl;
    //     uint8_t test[4];
    //     for (int i = 0; i < 4; i++)
    //     {
    //         uint8_t a = temp;
    //         temp = temp >> 8;
    //         test[i] = a;
    //     }
    //     for (int i = 3; i >= 0 ; i--)
    //     {
    //         infile << test[i];
    //     }
    // }
    // infile.close();

    // ifstream file ("samples.idx3-ubyte", ios::binary);
    // if (file.is_open())
    // {
    //     int magic_number=0;
    //     int number_of_images=0;
    //     int n_rows=0;
    //     int n_cols=0;
    //     file.read((char*)&magic_number,sizeof(magic_number));
    //     cout << "magic_number: " << magic_number << endl;
    //     magic_number= ReverseInt(magic_number);
    //     cout << "magic_number: " << magic_number << endl;

    //     file.read((char*)&number_of_images,sizeof(number_of_images));
    //     cout << "number_of_images: " << number_of_images << endl;
    //     number_of_images= ReverseInt(number_of_images);
    //     cout << "number_of_images: " << number_of_images << endl;
        
    //     file.read((char*)&n_rows,sizeof(n_rows));
    //     cout << "n_rows: " << n_rows << endl;
    //     n_rows= ReverseInt(n_rows);
    //     cout << "n_rows: " << n_rows << endl;
        
    //     file.read((char*)&n_cols,sizeof(n_cols));
    //     cout << "n_cols: " << n_cols << endl;
    //     n_cols= ReverseInt(n_cols);
    //     cout << "n_cols: " << n_cols << endl;
    // }
    // file.close();
    
    waitKey(0);
    return 0;
}

void SaveRoiImage(string sourceName, string saveName)
{
    // 1. 读取图片
    Mat srcImage = imread(sourceName);

    // 2. 转换为灰度图片
    Mat srcImage_gray;
    cvtColor(srcImage, srcImage_gray, CV_BGR2GRAY);

    // 3. 转换为二值图片
    Mat binaryImage;
    threshold(srcImage_gray, binaryImage, 127, 255, THRESH_BINARY);

    // 4. 寻找轮廓
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

    // 5. 绘制轮廓
    Mat drawing(binaryImage.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    drawContours( drawing, contours_poly, -1, Scalar(0), 1);
    rectangle( drawing, boundRect[1].tl(), boundRect[1].br(), Scalar(0), 2, 8, 0 );

    // 6. 根据最小轮廓把原图的字体截取下来
    Mat roi;
    srcImage(boundRect[1]).copyTo(roi);
    // cvtColor(roi, roi_gray, CV_BGR2GRAY);
    // threshold(roi_gray, roi_gray, 127, 255, THRESH_BINARY);
    cout << saveName << endl;
    imwrite(saveName, roi);

    srcImage.release();
    roi.release();
}

int GetBigOrLitterEndian(void)
{
    int result;

    union {
        short s;
        char c[sizeof(short)];
    } un;

    un.s = 0x0102;

    if (sizeof(short) == 2)
    {
        if (un.c[0] == 1 && un.c[1] == 2)
        {
            printf("big-endian\n");
            return 0;
        }
        else if (un.c[0] == 2 && un.c[1] == 1)
        {
            printf("little-endian\n");
            return 1;
        }
        else
            printf("unknow\n");
    }
    else
        printf("sizeof(short) = %d\n", (int)sizeof(short));

    return -1;
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