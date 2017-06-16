#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

#define NUMSIZE 2  
#define GAUSSKERN 3.5  
// #define PI 3.14159265358979323846  

#define MAXOCTAVES      4  
#define INITSIGMA       0.5
// #define SIGMA       sqrt(3)
#define SCALESPEROCTAVE 2

// #define CONTRAST_THRESHOLD   0.02  
// #define CURVATURE_THRESHOLD  10.0  
#define DOUBLE_BASE_IMAGE_SIZE 1  
// #define peakRelThresh 0.8  
// #define LEN 128

typedef struct ImageSt {        /*金字塔每一层*/  
    float levelsigma;  
    int levelsigmalength;  
    float absolute_sigma;  
    Mat Level;       //CvMat是OPENCV的矩阵类，其元素可以是图像的象素值         
} ImageLevels; 

typedef struct ImageSt1 {      /*金字塔每一阶梯*/  
    int row, col;          //Dimensions of image.   
    float subsample;  
    ImageLevels * Octave;                
} ImageOctaves;  

ImageOctaves *DOGoctaves;        
//DOG pyr，DOG算子计算简单，是尺度归一化的LoG算子的近似。  


//#define MYDEBUG
void MyDebug()
{
    #ifdef MYDEBUG 
    cout << "debug" << endl;
    waitKey(0);
    #endif
}

Mat ScaleInitImage(Mat & tempMat);
float* GaussianKernel1D(float sigma, int dim);
int BlurImage(Mat & src, Mat & dst, float sigma);
void Convolve1DWidth(float* kern, int dim, Mat & src, Mat & dst);
float ConvolveLocWidth(float* kernel, int dim, Mat & src, int x, int y);
void Convolve1DHeight(float* kern, int dim, Mat & src, Mat & dst);
float ConvolveLocHeight(float* kernel, int dim, Mat & src, int x, int y);
Mat doubleSizeImage2(Mat & im);
Mat halfSizeImage(Mat & im);
ImageOctaves* BuildGaussianOctaves(Mat & image);
Mat MosaicHorizen(Mat im1, Mat im2);
Mat MosaicVertical(Mat im1, Mat im2);

int numoctaves = 0;

int main()
{
    ImageOctaves *Gaussianpyr; 
    // 声明
    Mat src;
    Mat src_gray;
    
    // 1.读取图片
    src = imread("demo.jpg");
    //imshow("src", src);
    MyDebug();
    
    // 2.转为灰度图
    cvtColor(src, src_gray, CV_BGR2GRAY);
    imshow("src_gray", src_gray);
    MyDebug();

    // 3.归一化
    src_gray.convertTo(src_gray, CV_32FC1);
    src_gray /= 255.0;
    cout << "src_gray.dim: " << src_gray.dims << endl;

    // 4.计算金字塔阶数
    int dim = min(src_gray.rows, src_gray.cols);
    cout << "dim: " << dim << endl;
    numoctaves = (int) (log((double)dim) / log(2.0)) - 2;
    numoctaves = min(numoctaves, MAXOCTAVES);
    cout << "金字塔阶数为: " << numoctaves << endl;

    // 1.SIFT算法: 图像预处理，消除噪声，建立金字塔底层
    Mat tempMat = ScaleInitImage(src_gray);
    imshow("first step", tempMat);
    MyDebug();
    // 2.SIFT算法：建立高斯金字塔
    Gaussianpyr = BuildGaussianOctaves(tempMat);
    cout << "2.SIFT算法：建立高斯金字塔" << endl;

    cout << "**************拼接图片*************" << endl;
    //显示高斯金字塔  
    Mat mosaicHorizen1;
    Mat mosaicHorizen2;
    Mat mosaicVertical1;
    
    for (int i=0; i < numoctaves; i++)  
    {  
        if (i == 0)  
        {  
            mosaicHorizen1=MosaicHorizen( (Gaussianpyr[0].Octave)[0].Level, (Gaussianpyr[0].Octave)[1].Level);  
            for (int j=2;j<SCALESPEROCTAVE+3;j++)  
                mosaicHorizen1=MosaicHorizen( mosaicHorizen1, (Gaussianpyr[0].Octave)[j].Level );  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen1=halfSizeImage(mosaicHorizen1);  
        }  
        else if (i==1)  
        {  
            mosaicHorizen2=MosaicHorizen( (Gaussianpyr[1].Octave)[0].Level, (Gaussianpyr[1].Octave)[1].Level );  
            for (int j=2;j<SCALESPEROCTAVE+3;j++)  
                mosaicHorizen2=MosaicHorizen( mosaicHorizen2, (Gaussianpyr[1].Octave)[j].Level );  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen2=halfSizeImage(mosaicHorizen2);  
            mosaicVertical1=MosaicVertical( mosaicHorizen1, mosaicHorizen2 );  
        }  
        else  
        {  
            mosaicHorizen1=MosaicHorizen( (Gaussianpyr[i].Octave)[0].Level, (Gaussianpyr[i].Octave)[1].Level );  
            for (int j=2;j<SCALESPEROCTAVE+3;j++)  
                mosaicHorizen1=MosaicHorizen( mosaicHorizen1, (Gaussianpyr[i].Octave)[j].Level );  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen1=halfSizeImage(mosaicHorizen1);  
            mosaicVertical1=MosaicVertical( mosaicVertical1, mosaicHorizen1 );  
        }  
    }  
    imshow("mosaicVertical1", mosaicVertical1);
    // mosaic1 = cvCreateImage(cvSize(mosaicVertical1->width, mosaicVertical1->height),  IPL_DEPTH_8U,1);  
    // cvConvertScale( mosaicVertical1, mosaicVertical1, 255.0, 0 );  
    // cvConvertScaleAbs( mosaicVertical1, mosaic1, 1, 0 );  

    // 3.SIFT算法: 特征位置检测，确定特征点位置

    // 4.SIFT算法: 计算高斯图像的梯度方向和幅值，计算各个特征点主方向

    // 5.SIFT算法: 抽取各个特征点处的特征描述字

}

Mat MosaicVertical(Mat im1, Mat im2 )  
{   
    Mat mosaic = Mat::zeros(im1.rows+im2.rows, max(im1.cols,im2.cols), CV_32FC1);  

    /* Copy images into mosaic1. */  
    for (int row = 0; row < im1.rows; row++) {   
        for (int col = 0; col < im1.cols; col++) {
            mosaic.at<float>(row,col) = im1.at<float>(row,col);
        }  
    } 

    for (int row = 0; row < im2.rows; row++) {
        for (int col = 0; col < im2.cols; col++) {
            mosaic.at<float>((row+im1.rows),col) = im2.at<float>(row,col); 
        }  
    }  
 
    return mosaic;  
}  

Mat MosaicHorizen(Mat im1, Mat im2)  
{  
    Mat mosaic = Mat::zeros(max(im1.rows, im2.rows), (im1.cols + im2.cols), CV_32FC1);

    /* Copy images into mosaic1. */  
    for (int row = 0; row < im1.rows; row++) {
        for (int col = 0; col < im1.cols; col++) {
            mosaic.at<float>(row,col) = im1.at<float>(row,col) ;  
        }  
    }

    for (int row = 0; row < im2.rows; row++) {
        for (int col = 0; col < im2.cols; col++) {
            mosaic.at<float>(row, (col+im1.cols))= im2.at<float>(row,col) ;
        }
    }
                
    return mosaic;  
}  
  
ImageOctaves* BuildGaussianOctaves(Mat & image)
{
    ImageOctaves *octaves;

    double k = pow(2, 1.0/((float)SCALESPEROCTAVE));  //方差倍数  
    float initial_sigma, sigma, absolute_sigma, sigma_f;  

    //计算金字塔的阶梯数目  
    int dim = min(image.rows, image.cols);  
    int numoctaves = (int) (log((double) dim) / log(2.0)) - 2;    //金字塔阶数  

    //限定金字塔的阶梯数  
    numoctaves = min(numoctaves, MAXOCTAVES);  

    //为高斯金塔和DOG金字塔分配内存  
    octaves = (ImageOctaves*) malloc (numoctaves * sizeof(ImageOctaves));  
    DOGoctaves = (ImageOctaves*) malloc (numoctaves * sizeof(ImageOctaves));  

    printf("BuildGaussianOctaves(): Base image dimension is %dx%d\n", (int)(0.5*(image.cols)), (int)(0.5*(image.rows)) );  
    printf("BuildGaussianOctaves(): Building %d octaves\n", numoctaves);  

    // start with initial source image  
    Mat tempMat = image.clone();  
    // preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
    initial_sigma = sqrt(2);//sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );  
    //   initial_sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4);  

    //在每一阶金字塔图像中建立不同的尺度图像  
    for (int i = 0; i < numoctaves; i++)   
    {     
        //首先建立金字塔每一阶梯的最底层，其中0阶梯的最底层已经建立好  
        printf("Building octave %d of dimesion (%d, %d)\n", i, tempMat.cols,tempMat.rows);  
        //为各个阶梯分配内存  
        octaves[i].Octave = (ImageLevels*) malloc ((SCALESPEROCTAVE + 3) * sizeof(ImageLevels));  
        DOGoctaves[i].Octave = (ImageLevels*) malloc ((SCALESPEROCTAVE + 2) * sizeof(ImageLevels));
        cout << "内存分配成功" << endl;
        //存储各个阶梯的最底层  
        tempMat.copyTo((octaves[i].Octave)[0].Level);  
        cout << "tempMat" << endl;
        octaves[i].col = tempMat.cols;  
        octaves[i].row = tempMat.rows;  
        DOGoctaves[i].col = tempMat.cols;  
        DOGoctaves[i].row = tempMat.rows;

        if (DOUBLE_BASE_IMAGE_SIZE)  
            octaves[i].subsample = pow(2,i)*0.5;  
        else  
            octaves[i].subsample = pow(2,i);  
        
        cout << "a" << endl;

        if(i==0)       
        {  
            (octaves[0].Octave)[0].levelsigma = initial_sigma;  
            (octaves[0].Octave)[0].absolute_sigma = initial_sigma;  
            printf("0 scale and blur sigma : %f \n", (octaves[0].subsample) * ((octaves[0].Octave)[0].absolute_sigma));  
        }  
        else  
        {  
            (octaves[i].Octave)[0].levelsigma = (octaves[i-1].Octave)[SCALESPEROCTAVE].levelsigma;  
            (octaves[i].Octave)[0].absolute_sigma = (octaves[i-1].Octave)[SCALESPEROCTAVE].absolute_sigma;  
            printf( "0 scale and blur sigma : %f \n", ((octaves[i].Octave)[0].absolute_sigma) );  
        }  

        sigma = initial_sigma;  
        //建立本阶梯其他层的图像 
        cout << "建立本阶梯其他层的图像" << endl;
        for (int j =  1; j < SCALESPEROCTAVE + 3; j++)   
        {  
            Mat dst(tempMat.size(), CV_32FC1);//用于存储高斯层  
            //Mat temp(tempMat.size(), CV_32FC1);//用于存储DOG层  
            // 2 passes of 1D on original  
            //   if(i!=0)  
            //   {  
            //       sigma1 = pow(k, j - 1) * ((octaves[i-1].Octave)[j-1].levelsigma);  
            //          sigma2 = pow(k, j) * ((octaves[i].Octave)[j-1].levelsigma);  
            //       sigma = sqrt(sigma2*sigma2 - sigma1*sigma1);  
            sigma_f = sqrt(k*k-1)*sigma;  
            //   }  
            //   else  
            //   {  
            //       sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4)*pow(k,j);  
            //   }    
            sigma = k*sigma;  
            absolute_sigma = sigma * (octaves[i].subsample);  
            printf("%d scale and Blur sigma: %f  /n", j, absolute_sigma);  

            (octaves[i].Octave)[j].levelsigma = sigma;  
            (octaves[i].Octave)[j].absolute_sigma = absolute_sigma;  

            //产生高斯层  
            int length = BlurImage((octaves[i].Octave)[j-1].Level, dst, sigma_f);//相应尺度  
            (octaves[i].Octave)[j].levelsigmalength = length;  
            (octaves[i].Octave)[j].Level = dst.clone();  

            cout << "DOG层" << endl;
            //产生DOG层    
            Mat jMat = ((octaves[i].Octave)[j]).Level;
            Mat pjMat = ((octaves[i].Octave)[j-1]).Level;
            Mat temp = jMat - pjMat;
            //         cvAbsDiff( ((octaves[i].Octave)[j]).Level, ((octaves[i].Octave)[j-1]).Level, temp );  
            ((DOGoctaves[i].Octave)[j-1]).Level = temp.clone();  
        }  

        cout << "**********************end********************" << endl;
        // // halve the image size for next iteration  
        tempMat  = halfSizeImage(((octaves[i].Octave)[SCALESPEROCTAVE].Level)); 
        cout << "half" << endl;
    }  
    cout << "**********************return********************" << endl;
    return octaves;  
}

// SIFT算法第一步：扩大图像，预滤波剔除噪声，得到金字塔的最底层-第一阶的第一层
Mat ScaleInitImage(Mat & im)
{
    double sigma, preblur_sigma;
    Mat imMat(im.size(), CV_32FC1); 

    // 1.平滑滤波
    BlurImage(im, imMat, INITSIGMA);
    imshow("res", imMat);
    MyDebug();
    
    // 2.针对两种情况分别进行处理：初始化放大原始图像或者在原图像基础上进行后续操作  
    //建立金字塔的最底层  
    if (DOUBLE_BASE_IMAGE_SIZE)   
    {  
        Mat tempMat = doubleSizeImage2(imMat);  //对扩大两倍的图像进行二次采样，采样率为0.5，采用线性插值  
        Mat dst(tempMat.size(), CV_32FC1);

        preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
        BlurImage(tempMat, dst, preblur_sigma);   

        // The initial blurring for the first image of the first octave of the pyramid.  
        sigma = sqrt((4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma);  
        //  sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4);  
        //printf("Init Sigma: %f/n", sigma);  
        BlurImage(dst, tempMat, sigma);       //得到金字塔的最底层-放大2倍的图像  
        dst.release();

        return tempMat;  
    }   
    else   
    {  
        Mat dst(im.size(), CV_32FC1); 
        //sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA);  
        preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
        sigma = sqrt((4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma);  
        //printf("Init Sigma: %f/n", sigma);  
        BlurImage(imMat, dst, sigma);        //得到金字塔的最底层：原始图像大小  
        return dst;  
    }   
}

//上采样原来的图像，返回放大2倍尺寸的线性插值图像  
Mat doubleSizeImage2(Mat & im)   
{  
    Mat imnew(im.rows * 2, im.cols * 2, CV_32FC1);

    // fill every pixel so we don't have to worry about skipping pixels later  
    for (int j = 0; j < imnew.rows; j++)   
    {  
        for (int i = 0; i < imnew.cols; i++)   
        {  
            imnew.at<float>(j,i) = im.at<float>(j/2, i/2);  
        }  
    }  

    /* 
     *   A B C 
     *   E F G 
     *   H I J 
     *   pixels A C H J are pixels from original image 
     *   pixels B E G I F are interpolated pixels 
     */  
    // interpolate pixels B and I  
    for (int j = 0; j < imnew.rows; j += 2) {
        for (int i = 1; i < imnew.cols - 1; i += 2) {
            imnew.at<float>(j,i) = 0.5 * (im.at<float>(j/2, i/2) + im.at<float>(j/2, i/2+1));  
        } 
    } 
    
    // interpolate pixels E and G  
    for (int j = 1; j < imnew.rows - 1; j += 2) {
        for (int i = 0; i < imnew.cols; i += 2) {
            imnew.at<float>(j,i) = 0.5 * (im.at<float>(j/2, i/2) + im.at<float>(j/2+1, i/2));
        }
    } 
          
    // interpolate pixel F  
    for (int j = 1; j < imnew.rows - 1; j += 2) {
        for (int i = 1; i < imnew.cols - 1; i += 2) {
            imnew.at<float>(j,i) = 0.25 * (im.at<float>(j/2, i/2) + im.at<float>(j/2+1, i/2)
                                    + im.at<float>(j/2, i/2+1) + im.at<float>(j/2+1, i/2+1));
        }
    } 
      
    return imnew;  
}  

//下采样原来的图像，返回缩小2倍尺寸的图像  
Mat halfSizeImage(Mat & im)   
{  
    int w = im.cols/2;  
    int h = im.rows/2;   
    Mat imnew(h, w, CV_32FC1);  
 
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            imnew.at<float>(j,i) = im.at<float>(j*2, i*2);  
        }  
    }  
    
    return imnew;  
}  

// 卷积模糊图形
int BlurImage(Mat & src, Mat & dst, float sigma)
{
    cout << "BlurImage" << endl;

    float * convkernel;
    int dim = (int)max(3.0, 2.0 * GAUSSKERN * sigma + 1.0);  

    if (dim % 2 == 0)  
        dim++;  
    
    cout << "dim: " << dim << endl;
    Mat tempMat(src.size(), CV_32FC1); 
 
    convkernel = GaussianKernel1D(sigma, dim);  
    cout << "convkernel end" << endl;
    Convolve1DWidth(convkernel, dim, src, tempMat);  
    cout << "Convolve1DWidth end" << endl;
    Convolve1DHeight(convkernel, dim, tempMat, dst);  
    cout << "Convolve1DHeight end" << endl;
    tempMat.release();

    return dim;  
}

//产生1D高斯核  
float* GaussianKernel1D(float sigma, int dim)   
{  
    cout << "GaussianKernel1D" << endl;
    float *kern=(float*)malloc(dim * sizeof(float));  
    float s2 = sigma * sigma;  
    int c = dim / 2;  
    float m= 1.0 / (sqrt(2.0 * CV_PI) * sigma);  
    double v;   

    cout << "s2: " << s2 << " "
        << "c: " << c << " "
        << "m: " << m << endl;

    for (int i = 0; i < (dim + 1) / 2; i++)   
    {  
        v = m * exp(-(1.0*i*i)/(2.0 * s2)) ;  
        kern[c+i] = v;  
        kern[c-i] = v;  
    }  

    return kern;  
}

//x方向作卷积  
void Convolve1DWidth(float* kern, int dim, Mat & src, Mat & dst)   
{  
    //#define DST(ROW,COL) ((float *)(dst->data.fl + dst->step/sizeof(float) *(ROW)))[(COL)]  
    for (int j = 0; j < (int)src.rows; j++)   
    {  
        for (int i = 0; i < (int)src.cols; i++)   
        {   
            //DST(j,i) = ConvolveLocWidth(kern, dim, src, i, j);  
            dst.at<float>(j, i) = ConvolveLocWidth(kern, dim, src, i, j); 
        }  
    }  
}  

float ConvolveLocWidth(float* kernel, int dim, Mat & src, int x, int y)   
{  
    //#define Src(ROW,COL) ((float *)(src->data.fl + src->step/sizeof(float) *(ROW)))[(COL)]  
    float pixel = 0;  
    int col;  
    int cen = dim / 2;  
    
    //printf("ConvolveLoc(): Applying convoluation at location (%d, %d)/n", x, y);  
    for (int i = 0; i < dim; i++)   
    {  
        col = x + (i - cen);  
        if (col < 0)  
            col = 0;  
        if (col >= src.cols)  
            col = src.cols - 1;  
        //pixel += kernel[i] * Src(y,col); 
        pixel += kernel[i] * src.at<float>(y, col); 
    }  

    if (pixel > 1)  
        pixel = 1;  

    return pixel;  
}  

//y方向作卷积  
void Convolve1DHeight(float* kern, int dim, Mat & src, Mat & dst)   
{  
    cout << "src rows: " << src.rows << " col: " << src.cols << endl;
    // #define Dst(ROW,COL) ((float *)(dst->data.fl + dst->step/sizeof(float) *(ROW)))[(COL)]  
    for (int j = 0; j < (int)src.rows; j++)   
    {  
        for (int i = 0; i < (int)src.cols; i++)   
        {  
            // cout << "i: " << i << " j: " << j << endl;
            // Dst(j,i) = ConvolveLocHeight(kern, dim, src, i, j);  
            dst.at<float>(j, i) = ConvolveLocHeight(kern, dim, src, i, j);
        }  
    }  
}  

//y方向像素处作卷积  
float ConvolveLocHeight(float* kernel, int dim, Mat & src, int x, int y)   
{  
    //#define Src(ROW,COL) ((float *)(src->data.fl + src->step/sizeof(float) *(ROW)))[(COL)]  
    float pixel = 0;  
    int cen = dim / 2;  
    // printf("ConvolveLoc(): Applying convoluation at location (%d, %d)\n", x, y);  
    // cout << dim << endl;
    for (int j = 0; j < dim; j++)   
    {  
        int row = y + (j - cen);  
        // cout << row << " " << j << endl;
        if (row < 0)  
            row = 0;  
        if (row >= src.rows)  
            row = src.rows - 1;  

        pixel += kernel[j] * src.at<float>(row,x);  
    }  

    if (pixel > 1)  
        pixel = 1;   
    // cout << pixel << endl;
    return pixel;  
}  