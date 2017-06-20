#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

#define NUMSIZE 2  
#define GAUSSKERN 3.5  
#define PI 3.14159265358979323846  

#define MAXOCTAVES      4  
#define INITSIGMA       0.5
#define SIGMA       sqrt(3)
#define SCALESPEROCTAVE 2

#define CONTRAST_THRESHOLD   0.02  
#define CURVATURE_THRESHOLD  10.0  
#define DOUBLE_BASE_IMAGE_SIZE 1  
#define peakRelThresh 0.8  
#define LEN 128

#define GridSpacing 4 

typedef struct ImageSt {        /*金字塔每一层*/  
    float levelsigma;  
    int levelsigmalength;  
    float absolute_sigma;  
} ImageLevels; 

typedef struct ImageSt1 {      /*金字塔每一阶梯*/  
    int row, col;          //Dimensions of image.   
    float subsample;  
    ImageLevels * Octave;                
} ImageOctaves;  

typedef struct KeypointSt   
{  
    float row, col; /* 反馈回原图像大小，特征点的位置 */  
    float sx,sy;    /* 金字塔中特征点的位置*/  
    int octave,level;/*金字塔中，特征点所在的阶梯、层次*/  

    float scale, ori,mag; /*所在层的尺度sigma,主方向orientation (range [-PI,PI])，以及幅值*/  
    float *descrip;       /*特征描述字指针：128维或32维等*/  
    struct KeypointSt *next;/* Pointer to next keypoint in list. */  
} *Keypoint;  

//定义特征点具体变量  
Keypoint keypoints=NULL;      //用于临时存储特征点的位置等  
Keypoint keyDescriptors=NULL; //用于最后的确定特征点以及特征描述字  

ImageOctaves *DOGoctaves;   

ImageOctaves *mag_thresh;  
ImageOctaves *mag_pyr;  
ImageOctaves *grad_pyr; 

//DOG pyr，DOG算子计算简单，是尺度归一化的LoG算子的近似。  
vector<Mat> ImageLevelsGroup;
vector<Mat> DOGGroup;

vector<Mat> magThreshGroup;
vector<Mat> magpyrGroup;
vector<Mat> gradpyrGroup;

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
int DetectKeypoint(int numoctaves, ImageOctaves *GaussianPyr);
void DisplayKeypointLocation(Mat & image, ImageOctaves *GaussianPyr);
void ComputeGrad_DirecandMag(int numoctaves, ImageOctaves *GaussianPyr);
Mat GaussianKernel2D(float sigma);
int FindClosestRotationBin (int binCount, float angle);
void AverageWeakBins (double* hist, int binCount);
bool InterpolateOrientation (double left, double middle,double right, double *degreeCorrection, double *peakValue);
void AssignTheMainOrientation(int numoctaves, ImageOctaves *GaussianPyr,ImageOctaves *mag_pyr,ImageOctaves *grad_pyr);
void DisplayOrientation (Mat & image, ImageOctaves *GaussianPyr);
void ExtractFeatureDescriptors(int numoctaves, ImageOctaves *GaussianPyr);
float getPixelBI(Mat & im, float col, float row);
float GetVecNorm(float* vec, int dim);

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
    // imshow("src_gray", src_gray);
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
    // imshow("first step", tempMat);
    // MyDebug();
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
            // mosaicHorizen1=MosaicHorizen((Gaussianpyr[0].Octave)[0].Level, (Gaussianpyr[0].Octave)[1].Level); 
            mosaicHorizen1 = MosaicHorizen(ImageLevelsGroup.at(0), ImageLevelsGroup.at(1)); 
            for (int j=2;j<SCALESPEROCTAVE+3;j++)  
                mosaicHorizen1=MosaicHorizen(mosaicHorizen1, ImageLevelsGroup.at(j));  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen1=halfSizeImage(mosaicHorizen1);  
        }  
        else if (i==1)  
        {  
            mosaicHorizen2=MosaicHorizen(ImageLevelsGroup.at(0 + SCALESPEROCTAVE+3), ImageLevelsGroup.at(1 + SCALESPEROCTAVE+3));  
            for (int j=2;j<SCALESPEROCTAVE+3;j++)  
                mosaicHorizen2=MosaicHorizen(mosaicHorizen2, ImageLevelsGroup.at(j + SCALESPEROCTAVE+3));  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen2=halfSizeImage(mosaicHorizen2);  
            mosaicVertical1=MosaicVertical(mosaicHorizen1, mosaicHorizen2 ); 
        }  
        else  
        {  
            mosaicHorizen1=MosaicHorizen(ImageLevelsGroup.at(0 + i*(SCALESPEROCTAVE+3)), ImageLevelsGroup.at(1 + i*(SCALESPEROCTAVE+3)));  
            for (int j=2;j<SCALESPEROCTAVE+3;j++)  
                mosaicHorizen1=MosaicHorizen( mosaicHorizen1, ImageLevelsGroup.at(j + i*(SCALESPEROCTAVE+3)));  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen1=halfSizeImage(mosaicHorizen1);  
            mosaicVertical1=MosaicVertical( mosaicVertical1, mosaicHorizen1 );  
        }  
    }  
    imshow("mosaicVertical1", mosaicVertical1);
    waitKey(0);

    // 显示DOG金字塔
    for (int i=0; i<numoctaves;i++)  
    {  
        if (i==0)  
        {  
            mosaicHorizen1=MosaicHorizen(DOGGroup.at(0), DOGGroup.at(1));  
            for (int j=2;j<SCALESPEROCTAVE+2;j++)  
                mosaicHorizen1=MosaicHorizen(mosaicHorizen1, DOGGroup.at(j));  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen1=halfSizeImage(mosaicHorizen1);  
        }  
        else if (i==1)  
        {  
            mosaicHorizen2=MosaicHorizen(DOGGroup.at(0 + (SCALESPEROCTAVE+2)), DOGGroup.at(1 + (SCALESPEROCTAVE+2)));  
            for (int j=2;j<SCALESPEROCTAVE+2;j++)  
                mosaicHorizen2=MosaicHorizen(mosaicHorizen2, DOGGroup.at(j + (SCALESPEROCTAVE+2)));  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen2=halfSizeImage(mosaicHorizen2);  
            mosaicVertical1=MosaicVertical( mosaicHorizen1, mosaicHorizen2 );  
        }  
        else  
        {  
            mosaicHorizen1=MosaicHorizen(DOGGroup.at(0 + i*(SCALESPEROCTAVE+2)), DOGGroup.at(1 + i*(SCALESPEROCTAVE+2)));  
            for (int j=2;j<SCALESPEROCTAVE+2;j++)  
                mosaicHorizen1=MosaicHorizen( mosaicHorizen1, DOGGroup.at(j + i*(SCALESPEROCTAVE+2)));  
            for (int j=0;j<NUMSIZE;j++)  
                mosaicHorizen1=halfSizeImage(mosaicHorizen1);  
            mosaicVertical1=MosaicVertical( mosaicVertical1, mosaicHorizen1 );  
        }  
    }  
    mosaicVertical1 = ~mosaicVertical1;
    imshow("mosaicVertical1", mosaicVertical1);
    waitKey(0);

    // 3.SIFT算法: 特征位置检测，确定特征点位置
    int keycount=DetectKeypoint(numoctaves, Gaussianpyr);
    printf("the keypoints number are %d ;\n", keycount); 
    Mat image1 = src.clone();
    DisplayKeypointLocation(image1 ,Gaussianpyr); 

    // 4.SIFT算法: 计算高斯图像的梯度方向和幅值，计算各个特征点主方向
    ComputeGrad_DirecandMag(numoctaves, Gaussianpyr);  
    AssignTheMainOrientation(numoctaves, Gaussianpyr, mag_pyr, grad_pyr);
    Mat image2 = src.clone();  
    DisplayOrientation (image2, Gaussianpyr);
    imshow("img2", image2);
    waitKey(0);

    // 5.SIFT算法: 抽取各个特征点处的特征描述字
    ExtractFeatureDescriptors(numoctaves, Gaussianpyr);
}

//双线性插值，返回像素间的灰度值  
float getPixelBI(Mat & im, float col, float row)   
{  
    int irow, icol;  
    float rfrac, cfrac;  
    float row1 = 0, row2 = 0;  
    int width=im.cols;  
    int height=im.rows;  

    irow = (int) row;  
    icol = (int) col;  

    if (irow < 0 || irow >= height || icol < 0 || icol >= width)  
        return 0;  
    if (row > height - 1)  
        row = height - 1;  
    if (col > width - 1)  
        col = width - 1;  

    rfrac = 1.0 - (row - (float) irow);  
    cfrac = 1.0 - (col - (float) icol);  

    if (cfrac < 1)   
    {  
        row1 = cfrac * im.at<float>(irow,icol) + (1.0 - cfrac) * im.at<float>(irow,icol+1);  
    }   
    else   
    {  
        row1 = im.at<float>(irow,icol);  
    }  
    if (rfrac < 1)   
    {  
        if (cfrac < 1)   
        {  
            row2 = cfrac * im.at<float>(irow+1,icol) + (1.0 - cfrac) * im.at<float>(irow+1,icol+1);  
        } else   
        {  
            row2 = im.at<float>(irow+1,icol);  
        }  
    }  
    return rfrac * row1 + (1.0 - rfrac) * row2;  
}  

void ExtractFeatureDescriptors(int numoctaves, ImageOctaves *GaussianPyr)  
{  
    // The orientation histograms have 8 bins  
    float orient_bin_spacing = PI/4;  
    float orient_angles[8]={-PI,-PI+orient_bin_spacing,-PI*0.5, -orient_bin_spacing,
                        0.0, orient_bin_spacing, PI*0.5,  PI+orient_bin_spacing};  

    //产生描述字中心各点坐标  
    float *feat_grid=(float *) malloc( 2*16 * sizeof(float));  
    for (int i=0;i<GridSpacing;i++)  
    {  
        for (int j=0;j<2*GridSpacing;++j,++j)  
        {  
            feat_grid[i*2*GridSpacing+j]=-6.0+i*GridSpacing;  
            feat_grid[i*2*GridSpacing+j+1]=-6.0+0.5*j*GridSpacing;  
        }  
    }  

    //产生网格  
    float *feat_samples=(float *) malloc( 2*256 * sizeof(float));  
    for (int i=0;i<4*GridSpacing;i++)  
    {  
        for (int j=0;j<8*GridSpacing;j+=2)  
        {  
            feat_samples[i*8*GridSpacing+j]=-(2*GridSpacing-0.5)+i;  
            feat_samples[i*8*GridSpacing+j+1]=-(2*GridSpacing-0.5)+0.5*j;  
        }  
    }  

    float feat_window = 2*GridSpacing;  
    Keypoint p = keyDescriptors; // p指向第一个结点  
    while(p) // 没到表尾  
    {  
        // float scale=(GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;  
        float sine = sin(p->ori);  
        float cosine = cos(p->ori);    
        //计算中心点坐标旋转之后的位置  
        float *featcenter=(float *) malloc( 2*16 * sizeof(float));  

        for (int i=0;i<GridSpacing;i++)  
        {  
            for (int j=0;j<2*GridSpacing;j+=2)  
            {  
                float x=feat_grid[i*2*GridSpacing+j];  
                float y=feat_grid[i*2*GridSpacing+j+1];  
                featcenter[i*2*GridSpacing+j]=((cosine * x + sine * y) + p->sx);  
                featcenter[i*2*GridSpacing+j+1]=((-sine * x + cosine * y) + p->sy);  
            }  
        }  

        // calculate sample window coordinates (rotated along keypoint)  
        float *feat=(float *) malloc( 2*256 * sizeof(float));  
        for (int i=0;i<64*GridSpacing;i++,i++)  
        {  
            float x=feat_samples[i];  
            float y=feat_samples[i+1];  
            feat[i]=((cosine * x + sine * y) + p->sx);  
            feat[i+1]=((-sine * x + cosine * y) + p->sy);  
        }  

        //Initialize the feature descriptor.  
        float *feat_desc = (float *) malloc( 128 * sizeof(float));  
        for (int i=0;i<128;i++)  
        {  
            feat_desc[i]=0.0;  
            // printf("%f  ",feat_desc[i]);    
        }  
        //printf("/n");  
        for (int i=0;i<512;++i,++i)  
        {  
            float x_sample = feat[i];  
            float y_sample = feat[i+1];  
            // Interpolate the gradient at the sample position  
            /* 
            0   1   0 
            1   *   1 
            0   1   0   具体插值策略如图示 
            */  
            //((GaussianPyr[p->octave].Octave)[p->level]).Level
            float sample12=getPixelBI(ImageLevelsGroup.at(p->octave*(SCALESPEROCTAVE+3)+p->level), x_sample, y_sample-1);  
            float sample21=getPixelBI(ImageLevelsGroup.at(p->octave*(SCALESPEROCTAVE+3)+p->level), x_sample-1, y_sample);   
            // float sample22=getPixelBI(ImageLevelsGroup.at(p->octave*(SCALESPEROCTAVE+3)+p->level), x_sample, y_sample);   
            float sample23=getPixelBI(ImageLevelsGroup.at(p->octave*(SCALESPEROCTAVE+3)+p->level), x_sample+1, y_sample);   
            float sample32=getPixelBI(ImageLevelsGroup.at(p->octave*(SCALESPEROCTAVE+3)+p->level), x_sample, y_sample+1);   
            //float diff_x = 0.5*(sample23 - sample21);  
            //float diff_y = 0.5*(sample32 - sample12);  
            float diff_x = sample23 - sample21;  
            float diff_y = sample32 - sample12;  
            float mag_sample = sqrt( diff_x*diff_x + diff_y*diff_y );  
            float grad_sample = atan( diff_y / diff_x );  

            if(grad_sample == CV_PI)  
                grad_sample = -CV_PI;  
            // Compute the weighting for the x and y dimensions.  

            float *x_wght=(float *) malloc( GridSpacing * GridSpacing * sizeof(float));  
            float *y_wght=(float *) malloc( GridSpacing * GridSpacing * sizeof(float));  
            float *pos_wght=(float *) malloc( 8*GridSpacing * GridSpacing * sizeof(float));  

            for (int m=0;m<32;++m,++m)  
            {  
                float x=featcenter[m];  
                float y=featcenter[m+1];  
                x_wght[m/2] = max(1 - (fabs(x - x_sample)*1.0/GridSpacing), 0.0);  
                y_wght[m/2] = max(1 - (fabs(y - y_sample)*1.0/GridSpacing), 0.0);   
            }  

            for (int m=0;m<16;++m)  
                for (int n=0;n<8;++n)  
                pos_wght[m*8+n]=x_wght[m]*y_wght[m];  
            free(x_wght);  
            free(y_wght);  

            //计算方向的加权，首先旋转梯度场到主方向，然后计算差异   
            float diff[8],orient_wght[128];  
            for (int m=0;m<8;++m)  
            {   
                float angle = grad_sample-(p->ori)-orient_angles[m]+CV_PI;  
                float temp = angle / (2.0 * CV_PI);  
                angle -= (int)(temp) * (2.0 * CV_PI);  
                diff[m]= angle - CV_PI;  
            }  
            // Compute the gaussian weighting.  
            float x=p->sx;  
            float y=p->sy;  
            float g = exp(-((x_sample-x)*(x_sample-x)+(y_sample-y)*(y_sample-y))/(2*feat_window*feat_window))/(2*CV_PI*feat_window*feat_window);  

            for (int m=0;m<128;++m)  
            {  
                orient_wght[m] = max((1.0 - 1.0*fabs(diff[m%8])/orient_bin_spacing),0.0);  
                feat_desc[m] = feat_desc[m] + orient_wght[m]*pos_wght[m]*g*mag_sample;  
            }  
            free(pos_wght);     
        } 
        free(feat);  
        free(featcenter);  

        float norm=GetVecNorm( feat_desc, 128);  
        for (int m=0;m<128;m++)  
        {  
            feat_desc[m]/=norm;  
            if (feat_desc[m]>0.2)  
                feat_desc[m]=0.2;  
        }  
        norm=GetVecNorm( feat_desc, 128);  

        for (int m=0;m<128;m++)  
        {  
            feat_desc[m]/=norm;  
            printf("%f  ",feat_desc[m]);    
        }  
        printf("\n");  
        p->descrip = feat_desc;  
        p=p->next;  
    }  
    free(feat_grid);  
    free(feat_samples);  
}  

//得到向量的欧式长度，2-范数  
float GetVecNorm( float* vec, int dim )  
{  
    float sum=0.0;  

    for (int i=0;i<dim;i++)  
        sum+=vec[i]*vec[i]; 

    return sqrt(sum);  
}  

//在图像中，显示SIFT特征点的位置  
void DisplayKeypointLocation(Mat & image, ImageOctaves *GaussianPyr)  
{  
    Keypoint p = keypoints; // p指向第一个结点  
    while(p) // 没到表尾  
    {     
        Point pt1;
        pt1.x = (int)((p->col)-3);
        pt1.y = (int)(p->row);
        Point pt2;
        pt2.x = (int)((p->col)+3);
        pt2.y = (int)(p->row);
        line(image, pt1, pt2, Scalar(255,255,0), 1, 8, 0);
        
        pt1.x = (int)(p->col);
        pt1.y = (int)((p->row)-3);
        pt2.x = (int)(p->col);
        pt2.y = (int)((p->row)+3);
        line(image, pt1, pt2, Scalar(255,255,0), 1, 8, 0);
        
        p=p->next;  
    }   

    imshow("image", image);
    waitKey(0);
} 

//显示特征点处的主方向  
void DisplayOrientation (Mat & image, ImageOctaves *GaussianPyr)  
{  
    Point pt1;
    Point pt2;
    Keypoint p = keyDescriptors; // p指向第一个结点 

    while(p) // 没到表尾  
    {  
        float scale=(GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;  
        float autoscale = 3.0;   
        float uu=autoscale*scale*cos(p->ori);  
        float vv=autoscale*scale*sin(p->ori);  
        float x=(p->col)+uu;  
        float y=(p->row)+vv;  

        pt1.x = (int)(p->col);
        pt1.y = (int)(p->row);
        pt2.x = x;
        pt2.y = y;
        line(image, pt1, pt2, Scalar(255,255,0), 1, 8, 0);

        // Arrow head parameters  
        float alpha = 0.33; // Size of arrow head relative to the length of the vector  
        float beta = 0.33;  // Width of the base of the arrow head relative to the length  

        float xx0= (p->col)+uu-alpha*(uu+beta*vv);  
        float yy0= (p->row)+vv-alpha*(vv-beta*uu);  
        float xx1= (p->col)+uu-alpha*(uu-beta*vv);  
        float yy1= (p->row)+vv-alpha*(vv+beta*uu); 

        pt1.x = (int)xx0;
        pt1.y = (int)yy0;
        pt2.x = x;
        pt2.y = y;
        line(image, pt1, pt2, Scalar(255,255,0), 1, 8, 0);

        pt1.x = (int)xx1;
        pt1.y = (int)yy1;
        pt2.x = x;
        pt2.y = y;
        line(image, pt1, pt2, Scalar(255,255,0), 1, 8, 0);

        p=p->next;  
    }   
}  

// Compute the gradient direction and magnitude of the gaussian pyramid images  
void ComputeGrad_DirecandMag(int numoctaves, ImageOctaves *GaussianPyr)  
{  
    // ImageOctaves *mag_thresh ;  
    mag_pyr=(ImageOctaves*) malloc(numoctaves * sizeof(ImageOctaves) );  
    grad_pyr=(ImageOctaves*) malloc(numoctaves * sizeof(ImageOctaves) );  
    // float sigma=( (GaussianPyr[0].Octave)[SCALESPEROCTAVE+2].absolute_sigma ) / GaussianPyr[0].subsample;  
    // int dim = (int) (max(3.0f, 2 * GAUSSKERN *sigma + 1.0f)*0.5+0.5);  
    // #define ImLevels(OCTAVE,LEVEL,ROW,COL) ((float *)(GaussianPyr[(OCTAVE)].Octave[(LEVEL)].Level->data.fl + GaussianPyr[(OCTAVE)].Octave[(LEVEL)].Level->step/sizeof(float) *(ROW)))[(COL)]  
    for (int i=0; i<numoctaves; i++)    
    {          
        mag_pyr[i].Octave= (ImageLevels*) malloc( (SCALESPEROCTAVE) * sizeof(ImageLevels) );  
        grad_pyr[i].Octave= (ImageLevels*) malloc( (SCALESPEROCTAVE) * sizeof(ImageLevels) );  

        for(int j=1;j<SCALESPEROCTAVE+1;j++)//取中间的scaleperoctave个层  
        {    
            // CvMat *Mag = cvCreateMat(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);  
            // CvMat *Ori = cvCreateMat(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);  
            // CvMat *tempMat1 = cvCreateMat(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);  
            // CvMat *tempMat2 = cvCreateMat(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);  
            // cvZero(Mag);  
            // cvZero(Ori);  
            // cvZero(tempMat1);  
            // cvZero(tempMat2);   
            Mat Mag = Mat::zeros(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
            Mat Ori = Mat::zeros(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
            Mat tempMat1 = Mat::zeros(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
            Mat tempMat2 = Mat::zeros(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
            Mat tempLevels = ImageLevelsGroup.at(j+i*(SCALESPEROCTAVE+3));

            for (int m=1;m<(GaussianPyr[i].row-1);m++) 
            {
                for(int n=1;n<(GaussianPyr[i].col-1);n++)  
                {  
                    
                    //计算幅值  
                    tempMat1.at<float>(m,n) = 0.5*( tempLevels.at<float>(m,n+1)-tempLevels.at<float>(m,n-1) );  //dx  
                    tempMat2.at<float>(m,n) = 0.5*( tempLevels.at<float>(m+1,n)-tempLevels.at<float>(m-1,n) );  //dy  
                    Mag.at<float>(m,n) = sqrt(tempMat1.at<float>(m,n)*tempMat1.at<float>(m,n)+tempMat2.at<float>(m,n)*tempMat2.at<float>(m,n));  //mag  

                    //计算方向  
                    Ori.at<float>(m,n) =atan( tempMat2.at<float>(m,n)/tempMat1.at<float>(m,n) );  
                    if (Ori.at<float>(m,n) == CV_PI)  
                        Ori.at<float>(m,n) = -CV_PI;  
                }  
            }  
                
            // ((mag_pyr[i].Octave)[j-1]).Level=Mag;  
            // ((grad_pyr[i].Octave)[j-1]).Level=Ori; 
            magpyrGroup.push_back(Mag);
            gradpyrGroup.push_back(Ori);
            tempMat1.release();
            tempMat2.release();
        }//for levels  
    }//for octaves  
}  

//产生2D高斯核矩阵  
Mat GaussianKernel2D(float sigma)   
{  
    // int dim = (int) max(3.0f, GAUSSKERN * sigma);  
    int dim = (int) max(3.0, 2.0 * GAUSSKERN *sigma + 1.0);  

    // make dim odd  
    if (dim % 2 == 0)  
        dim++;  
    //printf("GaussianKernel(): Creating %dx%d matrix for sigma=%.3f gaussian/n", dim, dim, sigma);

    Mat mat(dim, dim, CV_32FC1);  
    float s2 = sigma * sigma;  
    int c = dim / 2;  
    //printf("%d %d/n", mat.size(), mat[0].size());  
    float m= 1.0/(sqrt(2.0 * CV_PI) * sigma);  
    for (int i = 0; i < (dim + 1) / 2; i++)   
    {  
        for (int j = 0; j < (dim + 1) / 2; j++)   
        {  
            //printf("%d %d %d/n", c, i, j);  
            float v = m * exp(-(1.0*i*i + 1.0*j*j) / (2.0 * s2));  
            mat.at<float>(c+i,c+j) = v;  
            mat.at<float>(c-i,c+j) = v;  
            mat.at<float>(c+i,c-j) = v;  
            mat.at<float>(c-i,c-j) = v;  
        }  
    }  
    // normalizeMat(mat);  
    return mat;  
}  

//寻找与方向直方图最近的柱，确定其index   
int FindClosestRotationBin (int binCount, float angle)  
{  
    angle += CV_PI;  
    angle /= 2.0 * CV_PI;  
    // calculate the aligned bin  
    angle *= binCount;  

    int idx = (int) angle;  
    if (idx == binCount)  
        idx = 0; 

    return (idx);  
}  

// Fit a parabol to the three points (-1.0 ; left), (0.0 ; middle) and  
// (1.0 ; right).  
// Formulas:  
// f(x) = a (x - c)^2 + b  
// c is the peak offset (where f'(x) is zero), b is the peak value.  
// In case there is an error false is returned, otherwise a correction  
// value between [-1 ; 1] is returned in 'degreeCorrection', where -1  
// means the peak is located completely at the left vector, and -0.5 just  
// in the middle between left and middle and > 0 to the right side. In  
// 'peakValue' the maximum estimated peak value is stored.  
bool InterpolateOrientation (double left, double middle,double right, double *degreeCorrection, double *peakValue)  
{  
    double a = ((left + right) - 2.0 * middle) / 2.0;   //抛物线捏合系数a  
    // degreeCorrection = peakValue = Double.NaN;  

    // Not a parabol  
    if (a == 0.0)  
        return false;  

    double c = (((left - middle) / a) - 1.0) / 2.0;  
    double b = middle - c * c * a;  
    if (c < -0.5 || c > 0.5)  
        return false;  

    *degreeCorrection = c;  
    *peakValue = b;  

    return true;  
}  

// Average the content of the direction bins.  
void AverageWeakBins (double* hist, int binCount)  
{  
    // TODO: make some tests what number of passes is the best. (its clear  
    // one is not enough, as we may have something like  
    // ( 0.4, 0.4, 0.3, 0.4, 0.4 ))  
    for (int sn = 0 ; sn < 2 ; ++sn)   
    {  
        double firstE = hist[0];  
        double last = hist[binCount-1];

        for (int sw = 0 ; sw < binCount ; ++sw)   
        {  
            double cur = hist[sw];  
            double next = (sw == (binCount - 1)) ? firstE : hist[(sw + 1) % binCount];  
            hist[sw] = (last + cur + next) / 3.0;  
            last = cur;  
        }  
    }  
}  

//SIFT算法第四步：计算各个特征点的主方向，确定主方向  
void AssignTheMainOrientation(int numoctaves, ImageOctaves *GaussianPyr,ImageOctaves *mag_pyr,ImageOctaves *grad_pyr)  
{
    // int num_bins = 36;  
    // float hist_step = 2.0*PI/num_bins;  
    // float hist_orient[36];  
    float sigma1=( ((GaussianPyr[0].Octave)[SCALESPEROCTAVE].absolute_sigma) ) / (GaussianPyr[0].subsample);//SCALESPEROCTAVE+2  
    int zero_pad = (int) (max(3.0, 2 * GAUSSKERN *sigma1 + 1.0)*0.5+0.5);  

    // for (int i=0;i<36;i++)  
    //     hist_orient[i]=-PI+i*hist_step;  
     
    // int keypoint_count = 0;  
    Keypoint p = keypoints; // p指向第一个结点  

    while (p)
    {
        int i=p->octave;  
        int j=p->level;  
        int m=p->sy;   //行  
        int n=p->sx;   //列  

        Mat tempLevels = ImageLevelsGroup.at(j+i*(SCALESPEROCTAVE+3));
        if ((m>=zero_pad)&&(m<GaussianPyr[i].row-zero_pad)&&(n>=zero_pad)&&(n<GaussianPyr[i].col-zero_pad))  
        {  
            float sigma=( ((GaussianPyr[i].Octave)[j].absolute_sigma) ) / (GaussianPyr[i].subsample);  

            //产生二维高斯模板  
            Mat mat = GaussianKernel2D( sigma );           
            int dim=(int)(0.5 * (mat.rows));  

            //分配用于存储Patch幅值和方向的空间  
            //#define MAT(ROW,COL) ((float *)(mat->data.fl + mat->step/sizeof(float) *(ROW)))[(COL)]  

            //声明方向直方图变量  
            double* orienthist = (double *) malloc(36 * sizeof(double));  
            for ( int sw = 0 ; sw < 36 ; ++sw)   
            {  
                orienthist[sw]=0.0;    
            }  

            //在特征点的周围统计梯度方向  
            for (int x=m-dim,mm=0;x<=(m+dim);x++,mm++)  
            {
                for(int y=n-dim,nn=0;y<=(n+dim);y++,nn++)  
                {       
                    //计算特征点处的幅值  
                    double dx = 0.5*(tempLevels.at<float>(x,y+1)-tempLevels.at<float>(x,y-1));  //dx  
                    double dy = 0.5*(tempLevels.at<float>(x+1,y)-tempLevels.at<float>(x-1,y));  //dy  
                    double mag = sqrt(dx*dx+dy*dy);  //mag  
                    //计算方向  
                    double Ori = atan( 1.0*dy/dx );  
                    int binIdx = FindClosestRotationBin(36, Ori);                   //得到离现有方向最近的直方块  
                    orienthist[binIdx] = orienthist[binIdx] + 1.0* mag * mat.at<float>(mm,nn);//利用高斯加权累加进直方图相应的块  
                }  
            } 
                
            // Find peaks in the orientation histogram using nonmax suppression.  
            AverageWeakBins (orienthist, 36);  
            // find the maximum peak in gradient orientation  
            double maxGrad = 0.0;  
            int maxBin = 0;  
            for (int b = 0 ; b < 36 ; ++b)   
            {  
                if (orienthist[b] > maxGrad)   
                {  
                    maxGrad = orienthist[b];  
                    maxBin = b;  
                }  
            }  

            // First determine the real interpolated peak high at the maximum bin  
            // position, which is guaranteed to be an absolute peak.  
            double maxPeakValue=0.0;  
            double maxDegreeCorrection=0.0;  
            if ( (InterpolateOrientation ( orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],  
                    orienthist[maxBin], orienthist[(maxBin + 1) % 36],  
                    &maxDegreeCorrection, &maxPeakValue)) == false)  
                printf("BUG: Parabola fitting broken");  

            // Now that we know the maximum peak value, we can find other keypoint  
            // orientations, which have to fulfill two criterias:  
            //  
            //  1. They must be a local peak themselves. Else we might add a very  
            //     similar keypoint orientation twice (imagine for example the  
            //     values: 0.4 1.0 0.8, if 1.0 is maximum peak, 0.8 is still added  
            //     with the default threshhold, but the maximum peak orientation  
            //     was already added).  
            //  2. They must have at least peakRelThresh times the maximum peak  
            //     value.  
            bool binIsKeypoint[36];  
            for (int b = 0 ; b < 36 ; ++b)   
            {  
                binIsKeypoint[b] = false;  
                // The maximum peak of course is  
                if (b == maxBin)   
                {  
                    binIsKeypoint[b] = true;  
                    continue;  
                }  

                // Local peaks are, too, in case they fulfill the threshhold  
                if (orienthist[b] < (peakRelThresh * maxPeakValue))  
                    continue; 

                int leftI = (b == 0) ? (36 - 1) : (b - 1);  
                int rightI = (b + 1) % 36;

                if (orienthist[b] <= orienthist[leftI] || orienthist[b] <= orienthist[rightI])  
                    continue; // no local peak  
                binIsKeypoint[b] = true;  
            } 

            // find other possible locations  
            double oneBinRad = (2.0 * PI) / 36;  
            for (int b = 0 ; b < 36 ; ++b)   
            {  
                if (binIsKeypoint[b] == false)  
                    continue;  
                // int bLeft = (b == 0) ? (36 - 1) : (b - 1);  
                // int bRight = (b + 1) % 36;  
                // Get an interpolated peak direction and value guess.  
                double peakValue;  
                double degreeCorrection;  

                // double maxPeakValue, maxDegreeCorrection;                
                if (InterpolateOrientation ( orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],  
                            orienthist[maxBin], orienthist[(maxBin + 1) % 36],  
                            &degreeCorrection, &peakValue) == false)  
                {  
                    printf("BUG: Parabola fitting broken");  
                }  

                double degree = (b + degreeCorrection) * oneBinRad - PI;  

                if (degree < -PI)  
                    degree += 2.0 * PI;  
                else if (degree > PI)  
                    degree -= 2.0 * PI;  

                //存储方向，可以直接利用检测到的链表进行该步主方向的指定;  
                //分配内存重新存储特征点  
                Keypoint k;  
                /* Allocate memory for the keypoint Descriptor. */  
                k = (Keypoint) malloc(sizeof(struct KeypointSt));  
                k->next = keyDescriptors;  
                keyDescriptors = k;  
                k->descrip = (float*)malloc(LEN * sizeof(float));  
                k->row = p->row;  
                k->col = p->col;  
                k->sy = p->sy;    //行  
                k->sx = p->sx;    //列  
                k->octave = p->octave;  
                k->level = p->level;  
                k->scale = p->scale;        
                k->ori = degree;  
                k->mag = peakValue;    
            }//for  
            free(orienthist);  
        }  
        p=p->next;  
    }
}

// 获取特征点
int DetectKeypoint(int numoctaves, ImageOctaves *GaussianPyr)  
{  
    //计算用于DOG极值点检测的主曲率比的阈值  
    double curvature_threshold= ((CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1))/CURVATURE_THRESHOLD;  
    int keypoint_count = 0;     
    
    for (int i = 0; i < numoctaves; i++)
    {
        for (int j = 0; j < SCALESPEROCTAVE+1; j++)
        {
            int dim = (int)(0.5*((GaussianPyr[i].Octave)[j].levelsigmalength)+0.5);  
            for (int m = dim; m < ((DOGoctaves[i].row)-dim); m++)
            {
                for(int n=dim;n<((DOGoctaves[i].col)-dim);n++)  
                {     
                    Mat tempMat = DOGGroup.at(j+i*(SCALESPEROCTAVE+2));
                    Mat ptempMat = DOGGroup.at(j-1+i*(SCALESPEROCTAVE+2));
                    Mat btempMat = DOGGroup.at(j+1+i*(SCALESPEROCTAVE+2));
                    float inf_val = tempMat.at<float>(m, n);
                    if(( (inf_val <= ptempMat.at<float>(m-1,n-1))&&  
                        (inf_val <= ptempMat.at<float>(m  ,n-1))&&  
                        (inf_val <= ptempMat.at<float>(m+1,n-1))&&  
                        (inf_val <= ptempMat.at<float>(m-1,n  ))&&  
                        (inf_val <= ptempMat.at<float>(m  ,n  ))&&  
                        (inf_val <= ptempMat.at<float>(m+1,n  ))&&  
                        (inf_val <= ptempMat.at<float>(m-1,n+1))&&  
                        (inf_val <= ptempMat.at<float>(m  ,n+1))&&  
                        (inf_val <= ptempMat.at<float>(m+1,n+1))&&  

                        (inf_val <= tempMat.at<float>(m-1,n-1))&&  
                        (inf_val <= tempMat.at<float>(m  ,n-1))&&  
                        (inf_val <= tempMat.at<float>(m+1,n-1))&&  
                        (inf_val <= tempMat.at<float>(m-1,n  ))&&  
                        (inf_val <= tempMat.at<float>(m+1,n  ))&&  
                        (inf_val <= tempMat.at<float>(m-1,n+1))&&  
                        (inf_val <= tempMat.at<float>(m  ,n+1))&&  
                        (inf_val <= tempMat.at<float>(m+1,n+1))&&     //当前层8  

                        (inf_val <= btempMat.at<float>(m-1,n-1))&&  
                        (inf_val <= btempMat.at<float>(m  ,n-1))&&  
                        (inf_val <= btempMat.at<float>(m+1,n-1))&&  
                        (inf_val <= btempMat.at<float>(m-1,n  ))&&  
                        (inf_val <= btempMat.at<float>(m  ,n  ))&&  
                        (inf_val <= btempMat.at<float>(m+1,n  ))&&  
                        (inf_val <= btempMat.at<float>(m-1,n+1))&&  
                        (inf_val <= btempMat.at<float>(m  ,n+1))&&  
                        (inf_val <= btempMat.at<float>(m+1,n+1))     //下一层大尺度9          
                        ) ||   
                        ( (inf_val >= ptempMat.at<float>(m-1,n-1))&&  
                        (inf_val >= ptempMat.at<float>(m  ,n-1))&&  
                        (inf_val >= ptempMat.at<float>(m+1,n-1))&&  
                        (inf_val >= ptempMat.at<float>(m-1,n  ))&&  
                        (inf_val >= ptempMat.at<float>(m  ,n  ))&&  
                        (inf_val >= ptempMat.at<float>(m+1,n  ))&&  
                        (inf_val >= ptempMat.at<float>(m-1,n+1))&&  
                        (inf_val >= ptempMat.at<float>(m  ,n+1))&&  
                        (inf_val >= ptempMat.at<float>(m+1,n+1))&&  
                        
                        (inf_val >= tempMat.at<float>(m-1,n-1))&&  
                        (inf_val >= tempMat.at<float>(m  ,n-1))&&  
                        (inf_val >= tempMat.at<float>(m+1,n-1))&&  
                        (inf_val >= tempMat.at<float>(m-1,n  ))&&  
                        (inf_val >= tempMat.at<float>(m+1,n  ))&&  
                        (inf_val >= tempMat.at<float>(m-1,n+1))&&  
                        (inf_val >= tempMat.at<float>(m  ,n+1))&&  
                        (inf_val >= tempMat.at<float>(m+1,n+1))&&     //当前层8  

                        (inf_val >= btempMat.at<float>(m-1,n-1))&&  
                        (inf_val >= btempMat.at<float>(m  ,n-1))&&  
                        (inf_val >= btempMat.at<float>(m+1,n-1))&&  
                        (inf_val >= btempMat.at<float>(m-1,n  ))&&  
                        (inf_val >= btempMat.at<float>(m  ,n  ))&&  
                        (inf_val >= btempMat.at<float>(m+1,n  ))&&  
                        (inf_val >= btempMat.at<float>(m-1,n+1))&&  
                        (inf_val >= btempMat.at<float>(m  ,n+1))&&  
                        (inf_val >= btempMat.at<float>(m+1,n+1))     //下一层大尺度9    
                        ) ) 
                    {
                        //此处可存储  
                        //然后必须具有明显的显著性，即必须大于CONTRAST_THRESHOLD=0.02  
                        if ( fabs(tempMat.at<float>(m, n)) >= CONTRAST_THRESHOLD )  
                        {
                            float Dxx,Dyy,Dxy,Tr_H,Det_H,curvature_ratio;  
                            Dxx = tempMat.at<float>(m,n-1) + tempMat.at<float>(m,n+1)-2.0*tempMat.at<float>(m,n);  
                            Dyy = tempMat.at<float>(m-1,n) + tempMat.at<float>(m+1,n)-2.0*tempMat.at<float>(m,n);  
                            Dxy = tempMat.at<float>(m-1,n-1) + tempMat.at<float>(m+1,n+1) - tempMat.at<float>(m+1,n-1) - tempMat.at<float>(m-1,n+1);  
                            Tr_H = Dxx + Dyy;  
                            Det_H = Dxx*Dyy - Dxy*Dxy;  
                            // Compute the ratio of the principal curvatures.  
                            curvature_ratio = (1.0*Tr_H*Tr_H)/Det_H;  

                            if ( (Det_H>=0.0) && (curvature_ratio <= curvature_threshold) )  //最后得到最具有显著性特征的特征点  
                            {  
                                //将其存储起来，以计算后面的特征描述字  
                                keypoint_count++;  
                                Keypoint k;  
                                /* Allocate memory for the keypoint. */  
                                k = (Keypoint)malloc(sizeof(struct KeypointSt));  
                                k->next = keypoints;  
                                keypoints = k;  
                                k->row = m*(GaussianPyr[i].subsample);  
                                k->col =n*(GaussianPyr[i].subsample);  
                                k->sy = m;    //行  
                                k->sx = n;    //列  
                                k->octave=i;  
                                k->level=j;  
                                k->scale = (GaussianPyr[i].Octave)[j].absolute_sigma;        
                            }//if >curvature_thresh  
                        }
                    }
                } 
            }
        }
    }
    return keypoint_count;  
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

    printf("BuildGaussianOctaves(): Base image dimension is %dx%d\n", (int)(0.5*(image.cols)), (int)(0.5*(image.rows)));  
    printf("BuildGaussianOctaves(): Building %d octaves\n", numoctaves);  

    // start with initial source image  
    Mat tempMat = image.clone();
    // preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
    initial_sigma = sqrt(2);//sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );  
    // initial_sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4);  

    cout << "numoctaves: " << numoctaves << " 存储金字塔不同尺度图像。。。。。。。。。。。。。" << endl;
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
        // tempMat.copyTo((octaves[i].Octave)[0].Level);  
        ImageLevelsGroup.push_back(tempMat);
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

            sigma_f = sqrt(k*k-1)*sigma;  

            sigma = k*sigma;  
            absolute_sigma = sigma * (octaves[i].subsample);  
            printf("%d scale and Blur sigma: %f  /n", j, absolute_sigma);  

            (octaves[i].Octave)[j].levelsigma = sigma;  
            (octaves[i].Octave)[j].absolute_sigma = absolute_sigma;  

            //产生高斯层  
            Mat tm = ImageLevelsGroup.at(j-1 + i*(SCALESPEROCTAVE + 3));
            int length = BlurImage(tm, dst, sigma_f);//相应尺度  
            (octaves[i].Octave)[j].levelsigmalength = length;  
            // (octaves[i].Octave)[j].Level = dst.clone();  
            ImageLevelsGroup.push_back(dst);
            cout << "DOG层" << endl;

            //产生DOG层    
            Mat jMat = ImageLevelsGroup.at(j + i*(SCALESPEROCTAVE + 3));
            Mat pjMat = ImageLevelsGroup.at(j-1 + i*(SCALESPEROCTAVE + 3));
            Mat temp = jMat.clone() - pjMat.clone();
            //         cvAbsDiff( ((octaves[i].Octave)[j]).Level, ((octaves[i].Octave)[j-1]).Level, temp );  
            DOGGroup.push_back(temp);
        }  

        cout << "**********************end********************" << endl;
        // // halve the image size for next iteration 

        // tempMat  = halfSizeImage((octaves[i].Octave)[SCALESPEROCTAVE].Level); 
        tempMat = halfSizeImage(ImageLevelsGroup.at(SCALESPEROCTAVE + i*(SCALESPEROCTAVE + 3)));
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
    // imshow("res", imMat);
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