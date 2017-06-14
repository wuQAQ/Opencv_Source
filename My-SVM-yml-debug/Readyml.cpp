#include "Readyml.h"

static string samplePrefix = "Mat_";
static int n_fea = 25;

Mat read_sample_yml(string fileName, int singleSample)
{
    Mat dataMat = Mat::zeros(singleSample*10, n_fea, CV_32FC1);
    FileStorage readSamplefs(fileName, FileStorage::READ);

    if (readSamplefs.isOpened())
    {
        for (int i = 0; i < 10; i++)
        {
            Mat temp;
            string mat_i = samplePrefix + to_string(i);
            readSamplefs[mat_i] >> temp;
            for (int j = 0; j < singleSample; j++)
            {
                Mat tempT = temp.t();
                dataMat.row(i + j*(singleSample)) += tempT.row(j);
            }
        }
    }
    readSamplefs.release();

    return dataMat;
}

Mat read_label_yml(int singleSample)
{
    Mat LabelMat = Mat::zeros(singleSample*10, 1, CV_32SC1);
    
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < singleSample; j++)
        {
            LabelMat.at<unsigned int>(j + i * singleSample) = i;
        }
    }

    return LabelMat;
}

Mat ChangePost(Mat & tempMat, vector<int> &randArray)
{
    Mat newMat = Mat::zeros(tempMat.rows, tempMat.cols, tempMat.type());

    for (int i = 0; i < (int)randArray.size(); i++)
    {
        int temp = randArray.at(i);
        newMat.row(i) += tempMat.row(temp); 
    }

    return newMat;
}