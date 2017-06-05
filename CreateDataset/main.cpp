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

#define SAMROWS   28
#define SAMCOLS   28
#define NEWSIZE   28

int GetBigOrLitterEndian(void);
vector<dirent> showAllFiles (const char * dir_name);
void SaveRoiImage(string sourceName, string saveName);
void WriteMagicNumber(int magicNumber, int byteorder, vector<uint8_t> & instream);
void GetSingleImageFeature(string name, vector<uint8_t> & features);

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

    // 7.判断主机字节序
    int byteorder = GetBigOrLitterEndian();
    if (0 == byteorder)
    {
        cout << "此程序不适合大端字节数的主机" << endl;
        exit(-1);
    }
    
    // 8.制作二进制文件
    ofstream inSample("samples.idx3-ubyte", ios::binary);
    ofstream inLabel("labels.idx1-ubyte", ios::binary);
    if (inSample.is_open() && inLabel.is_open())
    {
        vector<uint8_t> s_magicStream;
        vector<uint8_t> l_magicStream;
        int s_magicNumber = 0x0803;
        int l_magicNumber = 0x0801;
        int numberOfImages = 100;
        int numberOfRows = SAMROWS;
        int numberOfCols = SAMCOLS;

        WriteMagicNumber(s_magicNumber, byteorder, s_magicStream);
        WriteMagicNumber(numberOfImages, byteorder, s_magicStream);
        WriteMagicNumber(numberOfRows, byteorder, s_magicStream);
        WriteMagicNumber(numberOfCols, byteorder, s_magicStream);
        for (int i = 0; i < (int)s_magicStream.size(); i++)
        {
            uint8_t temp;
            temp = s_magicStream.at(i);
            inSample << temp;
        }

        WriteMagicNumber(l_magicNumber, byteorder, l_magicStream);
        WriteMagicNumber(numberOfImages, byteorder, l_magicStream);
        for (int i = 0; i < (int)l_magicStream.size(); i++)
        {
            uint8_t temp;
            temp = l_magicStream.at(i);
            inLabel << temp;
        }

        for (int i = 0; i < 10; i++)
        {
            uint8_t tempLabel = i;
            
            for (int j = 0; j < 10; j++)
            {
                inLabel << tempLabel;
                string filename = saveDir + to_string(i) + "-" + to_string(j) + ".jpg";
                cout << filename << endl;
                vector<uint8_t> features;
                GetSingleImageFeature(filename, features);

                for (int k = 0; k < (int)features.size(); k++)
                    inSample << features.at(k);

                features.clear();
                features.shrink_to_fit();
            }
        }
    }
    inSample.close();
    inLabel.close();

    ifstream file ("samples.idx3-ubyte", ios::binary);
    ifstream l_file ("labels.idx1-ubyte", ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int l_magic_number=0;
        int number_of_images=0;
        int l_number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        cout << "magic_number: " << magic_number << endl;

        l_file.read((char*)&l_magic_number,sizeof(l_magic_number));
        l_magic_number= ReverseInt(l_magic_number);
        cout << "magic_number: " << l_magic_number << endl;

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        cout << "number_of_images: " << number_of_images << endl;
        
        l_file.read((char*)&l_number_of_images,sizeof(l_number_of_images));
        l_number_of_images= ReverseInt(l_number_of_images);
        cout << "number_of_images: " << l_number_of_images << endl;

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        cout << "n_rows: " << n_rows << endl;
        
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        cout << "n_cols: " << n_cols << endl;

        for (int index = 0; index < number_of_images; index++)
        {
            uint8_t temp_label;

            l_file.read((char*)&temp_label, sizeof(temp_label));
            cout << (int)temp_label << endl;
            for (int i = 0; i < n_rows; i++)
            {
                for (int j = 0; j < n_cols; j++)
                {
                    uint8_t temp;
                    file.read((char*)&temp,sizeof(temp));
                    cout << (int)temp;
                }
                cout << endl;
            }
        }
    }
    file.close();
    l_file.close();

    waitKey(0);
    return 0;
}

void GetSingleImageFeature(string name, vector<uint8_t> & features)
{
    // 1. 读取文件
    Mat srcImage = imread(name);

    // 2. 转换为灰度图
    Mat srcImage_gray;
    cvtColor(srcImage, srcImage_gray, CV_BGR2GRAY);

    // 3. 转换为二值图
    Mat binaryImage;
    threshold(srcImage_gray, binaryImage, 127, 255, THRESH_BINARY);

    // 4. 缩放
    Mat newImage;
    resize(binaryImage, newImage, Size(NEWSIZE, NEWSIZE), 0, 0, 3); 

    // 5. 归一化
    newImage = ~newImage/255;
    
    // 6. 提取
    for (int i = 0; i < newImage.rows; i++)
    {
        for (int j = 0; j < newImage.cols; j++)
        {
            cout << (int)newImage.at<uint8_t>(i, j);
            features.push_back(newImage.at<uint8_t>(i, j));
        }
        cout << endl;
    }
}

void WriteMagicNumber(int magicNumber, int byteorder, vector<uint8_t> & instream)
{
    if (1 == byteorder)
    {
        uint8_t test[4];
        for (int i = 0; i < 4; i++)
        {
            uint8_t a = magicNumber;
            magicNumber = magicNumber >> 8;
            test[i] = a;
        }
        for (int i = 3; i >= 0 ; i--)
        {
            instream.push_back(test[i]);
        }
    }
    else if (0 == byteorder)
    {

    }
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

    // 4. 滤波
    medianBlur(binaryImage, binaryImage, 5);

    // 5. 腐蚀
    Mat element = getStructuringElement(MORPH_RECT, Size(20, 20/*15, 15*/));  
    erode(binaryImage, binaryImage,element);  
    // imshow("binary", binaryImage);

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
    // imshow("roi", roi);

    // 8. 保存
    cout << saveName << endl;
    imwrite(saveName, roi);

    srcImage.release();
    srcImage_gray.release();
    binaryImage.release();
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