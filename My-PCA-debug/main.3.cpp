#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
using namespace cv;  
using namespace std;  
  
void DoPca(const Mat &_data, int dim, Mat &eigenvalues, Mat &eigenvectors);  
   
void printMat( Mat _data )  
{  
    Mat data = cv::Mat_<double>(_data);  
    for ( int i=0; i<data.rows; i++ )  
    {  
        for ( int j=0; j< data.cols; j++ )  
        {  
            cout << data.at<double>(i,j) << "  ";  
        }  
        cout << endl;  
    }  
}  
   
int main(int argc, char* argv[])  
{  
    float A[ 60 ]={   
    1.5 , 2.3 , 1.5 , 2.3 , 1.5 , 2.3 ,   
    3.0 , 1.7 , 3.0 , 1.7 , 3.0 , 1.7 ,   
    1.2 , 2.9 , 1.2 , 2.9 , 1.2 , 2.9 ,   
    2.1 , 2.2 , 2.1 , 2.2 , 2.1 , 2.2 ,  
    3.1 , 3.1 , 3.1 , 3.1 , 3.1 , 3.1 ,   
    1.3 , 2.7 , 1.3 , 2.7 , 1.3 , 2.7 ,   
    2.0 , 1.7 , 2.0 , 1.7 , 2.0 , 1.7 ,   
    1.0 , 2.0 , 1.0 , 2.0 , 1.0 , 2.0 ,   
    0.5 , 0.6 , 0.5 , 0.6 , 0.5 , 0.6 ,   
    1.0 , 0.9 , 1.0 , 0.9 , 1.0 , 0.9 };   
   
    Mat DataMat = Mat::zeros( 10, 6, CV_32F );  
   
    //将数组A里的数据放入DataMat矩阵中  
    for ( int i=0; i<10; i++ )  
    {  
        for ( int j=0; j<6; j++ )  
        {  
            DataMat.at<float>(i, j) = A[i * 6 + j];  
        }  
    }  
   
    // OPENCV PCA  
    PCA pca(DataMat, noArray(), CV_PCA_DATA_AS_ROW);  
   
    Mat eigenvalues;//特征值  
    Mat eigenvectors;//特征向量  
   
    DoPca(DataMat, 3, eigenvalues, eigenvectors);  
   
    cout << "eigenvalues:" << endl;  
    printMat( eigenvalues );  
    cout << "\n" << endl;  
    cout << "eigenvectors:" << endl;  
    printMat( eigenvectors );  
   
    system("pause");  
    return 0;  
}  
   
   
void DoPca(const Mat &_data, int dim, Mat &eigenvalues, Mat &eigenvectors)  
{  
    assert( dim>0 );  
    Mat data =  cv::Mat_<double>(_data);  
   
    int R = data.rows;  
    int C = data.cols;  
   
    if ( dim>C )  
        dim = C;  
  
    //计算均值  
    Mat m = Mat::zeros( 1, C, data.type() );  
   
    for ( int j=0; j<C; j++ )  
    {  
        for ( int i=0; i<R; i++ )  
        {  
            m.at<double>(0,j) += data.at<double>(i,j);  
        }  
    }  
  
    m = m/R;   
    //求取6列数据对应的均值存放在m矩阵中，均值： [1.67、2.01、1.67、2.01、1.67、2.01]  
      
  
    //计算协方差矩阵  
    Mat S =  Mat::zeros( R, C, data.type() );  
    for ( int i=0; i<R; i++ )  
    {  
        for ( int j=0; j<C; j++ )  
        {  
            S.at<double>(i,j) = data.at<double>(i,j) - m.at<double>(0,j); // 数据矩阵的值减去对应列的均值  
        }  
    }  
      
    Mat Average = S.t() * S /(R);  
    //计算协方差矩阵的方式----(S矩阵的转置 * S矩阵)/行数  
  
  
    //使用opencv提供的eigen函数求特征值以及特征向量  
    eigen(Average, eigenvalues, eigenvectors);  
}  