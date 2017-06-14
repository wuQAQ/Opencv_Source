#include "svm.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

svm_parameter param;
svm_problem prob;

//计时器
double cost_time;
clock_t start_time;
clock_t end_time;

string trainImage = "mnist_dataset/train-images.idx3-ubyte";
string trainLabel = "mnist_dataset/train-labels.idx1-ubyte";
string testImage = "mnist_dataset/t10k-images.idx3-ubyte";
string testLabel = "mnist_dataset/t10k-labels.idx1-ubyte";

int reverseInt(int i);
void read_mnist_image(const string fileName);
void read_mnist_label(const string fileName);

void init_param()
{
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.0001;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 10;
	param.eps = 1e-5;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
}

void init_setting(void)
{
	int probfeature = 784;
	prob.l = 60000;
	prob.y = new double[prob.l];
	prob.x = new svm_node * [prob.l];
}

int main(){
	init_param();
    init_setting();
	double d;

	if(param.gamma == 0) param.gamma = 0.5;
	read_mnist_image(trainImage);
    read_mnist_label(trainLabel);

    svm_node *t = prob.x[0];
    for (int i = 0; i < 785; i++)
    {
        cout << t[i].index << ":" << t[i].value << endl;
    }

	delete[] prob.x;
	delete[] prob.y;
}

int reverseInt(int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_image(const string fileName) 
{
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    ifstream file(fileName.c_str(), ios::binary);
    if (file.is_open())
    {
        cout << "成功打开图像集 ... \n";

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));
        //cout << magic_number << " " << number_of_images << " " << n_rows << " " << n_cols << endl;

        magic_number = reverseInt(magic_number);
        number_of_images = reverseInt(number_of_images);
        n_rows = reverseInt(n_rows);
        n_cols = reverseInt(n_cols);
        cout << "MAGIC NUMBER = " << magic_number
            << " ;NUMBER OF IMAGES = " << number_of_images
            << " ; NUMBER OF ROWS = " << n_rows
            << " ; NUMBER OF COLS = " << n_cols << endl;

        //-test-
        //number_of_images = testNum;
        //输出第一张和最后一张图，检测读取数据无误
        cout << "开始读取Image数据......\n";
        start_time = clock();
   
        for (int i = 0; i < number_of_images; i++) {
            svm_node tempNode[n_rows * n_cols+1];
            for (int j = 0; j < n_rows * n_cols; j++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                double pixel_value = double((temp + 0.0) / 255.0);
                tempNode[j].index = j;
                tempNode[j].value = pixel_value;
            }
            tempNode[n_rows*n_cols].index = -1;
            prob.x[i] = tempNode;
            cout << i << endl;
        }
        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Image数据完毕......" << cost_time << "s\n";
    }
    file.close();
}

void read_mnist_label(const string fileName) 
{
    int magic_number;
    int number_of_items;

    ifstream file(fileName.c_str(), ios::binary);
    if (file.is_open())
    {
        cout << "成功打开Label集 ... \n";

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_items, sizeof(number_of_items));
        magic_number = reverseInt(magic_number);
        number_of_items = reverseInt(number_of_items);

        cout << "MAGIC NUMBER = " << magic_number << "  ; NUMBER OF ITEMS = " << number_of_items << endl;

        //-test-
        //number_of_items = testNum;
        //记录第一个label和最后一个label
        unsigned int s = 0, e = 0;

        cout << "开始读取Label数据......\n";
        start_time = clock();
        
        for (int i = 0; i < number_of_items; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            int tLabel = (int)temp;
            prob.y[i] = (double)temp;
            //cout << i << endl;
        }

        end_time = clock();
        cost_time = (end_time - start_time) / CLOCKS_PER_SEC;
        cout << "读取Label数据完毕......" << cost_time << "s\n";

        cout << "first label = " << s << endl;
        cout << "last label = " << e << endl;
    }
    file.close();
}