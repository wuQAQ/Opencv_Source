#include <iostream>
#include <fstream>
//#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

// Training image file name
const string training_image_fn = "samples.idx3-ubyte";

// Training label file name
const string training_label_fn = "labels.idx1-ubyte";

// Weights file name
const string model_fn = "my-model.dat";

// Report file name
const string report_fn = "my-training-report.dat";

// Number of training samples
const int nTraining = 100;
const int nTrainingTimes = 100;

// Image size in MNIST database
const int width = 20;
const int height = 20;

const int n1 = width * height; 
const int n2 = 100;
const int nl2 = 1;
const int n3 = 10; 

const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

typedef struct inputNode
{
    double outValue;
    vector<double> weight;
    vector<double> delta;
}inputNode;

typedef struct hiddenNode   // 隐含层节点
{
    double inValue;
    double outValue;
    double theta;
    vector<double> weight;
    vector<double> delta;
}hiddenNode;

typedef struct outputNode   // 输出层节点
{
    double inValue;
    double outValue;
    double theta;
    double expected;
}outputNode;

inputNode* inputLayer[n1]; 
hiddenNode* hiddenLayer[nl2][n2];
outputNode* outputLayer[n3];
int d[width + 1][height + 1];

ifstream image;
ifstream label;
ofstream report;

void InitNN(void)
{
    // 1.分配空间
    for (int i = 0; i < n1; i++)
    {
        inputLayer[i] = new inputNode();
    }

    for (int i = 0; i < nl2; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            hiddenLayer[i][j] = new hiddenNode();
        }
    }

    for (int i = 0; i < n3; i++)
    {
        outputLayer[i] = new outputNode();
    }

    // 2.初始化参数
    // 输入层->隐藏层的权重
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            int sign = rand() % 2;
            double temp = (double)(rand()%6)/10.0;
            if (sign == 1)
                temp = -temp;
            inputLayer[i]->weight.push_back(temp);
            inputLayer[i]->delta.push_back(0.0);
        }
    }
    
    //隐藏层->输出层
    for (int layer = 0; layer < nl2; layer++)
    {
        for (int i = 0; i < n2; i++)
        {
            int tempNum = 0;
            if (layer == (nl2 - 1))
                tempNum = n3;
            else 
                tempNum = n2;

            for (int j = 0; j < tempNum; j++)
            {
                int sign = rand() % 2;
                double temp = (double)(rand() % 10 + 1) / (10.0 * n3);
                
                if (sign == 1)
                    temp = -temp;
                hiddenLayer[layer][i]->weight.push_back(temp);
                hiddenLayer[layer][i]->delta.push_back(temp);
            }
        }
    }
}

// Sigmoid function
double sigmoid(double x) 
{
    return (1.0 / (1.0 + exp(-x)));
}

// Forward process - Perceptron
void Perceptron() 
{
    for (int layer = 0; layer < nl2; layer++)
    {
        for (int i = 0; i < n2; i++)
        {
            hiddenLayer[layer][i]->inValue = 0.0;
        }
    }

    for (int i = 0; i < n3; i++)
    {
        outputLayer[i]->inValue = 0.0;
    }

    // 隐藏层
    for (int layer = 0; layer < nl2; layer++)
    {
        if (layer == 0)
        {
            // 计算输入
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    hiddenLayer[layer][j]->inValue += 
                        inputLayer[i]->outValue *  inputLayer[i]->weight[j];
                }
            }
        }
        else
        {
            for (int i = 0; i < n2; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    hiddenLayer[layer][j]->inValue += 
                        inputLayer[i]->outValue *  inputLayer[i]->weight[j];
                }
            }
        }

        // 求输出
        for (int i = 0; i < n2; i++)
        {
            double temp = hiddenLayer[layer][i]->inValue;
            hiddenLayer[layer][i]->outValue = sigmoid(temp);
        }
    }

    // 输出层
    for (int i = 0; i < n2; i++)
    {
        for (int j = 0; j < n3; j++)
        {
            outputLayer[j]->inValue += 
                hiddenLayer[nl2-1][i]->outValue * hiddenLayer[nl2-1][i]->weight[j];
        }
    }

    for (int i = 0; i < n3; i++)
    {
        double temp = outputLayer[i]->inValue;
        outputLayer[i]->outValue = sigmoid(temp);
    } 
}

// 输出误差
double square_error()
{
    double res = 0.0;

    for (int i = 0; i < n3; i++)
    {
        double outTemp = outputLayer[i]->outValue;
        double expectedTemp = outputLayer[i]->expected;
        res += (outTemp - expectedTemp) * (outTemp - expectedTemp);
    }

    res *= 0.5;
    return res;
}

// Back Propagation Algorithm
void back_propagation()
{
    double sum;

    // 1.计算delta
    // 输出层
    for (int i = 0; i < n3; i++)
    {
        double outValueTemp = outputLayer[i]->outValue;
        double expectedTemp = outputLayer[i]->expected;
        outputLayer[i]->theta = 
            outValueTemp * (1-outValueTemp) * (expectedTemp - outValueTemp);
    }

    // 隐藏层
    for (int layer = (nl2-1); layer >= 0; layer--)
    {
        sum = 0.0;
        for (int i = 0; i < n2; i++)
        {   
            if (layer == (nl2-1))
            {   
                //  tempNum = n3;
                for (int j = 0; j < n3; j++)
                {
                    sum += hiddenLayer[layer][i]->weight[j] * outputLayer[j]->theta;
                }
            }
            else
            {
                // tempNum = n2;
                for (int j = 0; j < n2; j++)
                {
                    sum += hiddenLayer[layer][i]->weight[j] * outputLayer[i]->theta;
                }
            }

            //cout << layer << " " << i << "sum: " << sum << endl;
            double outValueTemp = hiddenLayer[layer][i]->outValue;
            hiddenLayer[layer][i]->theta = outValueTemp * (1-outValueTemp) * sum;
        }
        
    }

    //cout << "delta hidden" << endl;

    // 2.更新weight
    // 隐藏层
    for (int layer = (nl2-1); layer >= 0; layer--)
    {
        if (layer == (nl2-1))
        {
            for (int i = 0; i < n2; i++)
            {
                for (int j = 0; j < n3; j++)
                {
                    double thetaTemp = outputLayer[j]->theta;
                    double h_deltaTemp = hiddenLayer[layer][i]->delta[j];
                    double h_outValueTemp = hiddenLayer[layer][i]->outValue;
                    hiddenLayer[layer][i]->delta[j] =
                         (learning_rate * thetaTemp * h_outValueTemp) + (momentum * h_deltaTemp);
                    hiddenLayer[layer][i]->weight[j] += hiddenLayer[layer][i]->delta[j];
                }
            }
        }
        else
        {
            for (int i = 0; i < n2; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    double thetaTemp = hiddenLayer[layer+1][j]->theta;
                    double h_deltaTemp = hiddenLayer[layer][i]->delta[j];
                    double h_outValueTemp = hiddenLayer[layer][i]->outValue;
                    hiddenLayer[layer][i]->delta[j] =
                         (learning_rate * thetaTemp * h_outValueTemp) + (momentum * h_deltaTemp);
                    hiddenLayer[layer][i]->weight[j] += hiddenLayer[layer][i]->delta[j];
                }
            }
        }
    }
    
    // 输入层
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            double h_thetaTemp = hiddenLayer[0][j]->theta;
            double i_outValueTemp = inputLayer[i]->outValue;
            double i_deltaTemp = inputLayer[i]->delta[j];
            inputLayer[i]->delta[j] = 
                (learning_rate * h_thetaTemp * i_outValueTemp) + (momentum * i_deltaTemp);
        }
    }
    
}

int learning_process()
{
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            inputLayer[i]->delta[j] = 0.0;
        }
    }

    for (int layer = 0; layer < nl2; layer++)
    {
        for (int i = 0; i < n2; i++)
        {
            int tempNum;
            if (layer = (nl2 - 1))
                tempNum = n3;
            else
                tempNum = n2;
            
            for (int j = 0; j < tempNum; j++)
            {
                hiddenLayer[layer][i]->delta[j] = 0.0;
            }
        }
    }

    //cout << "p_init end" << endl;
    for (int i = 0; i < epochs; i++)
    {
        Perceptron();
        //cout << "Perceptron" << endl;
        back_propagation();
        //cout << "back_propagation" << endl;
        if (square_error() < epsilon) {
			return i;
		}
    }
    return epochs;
}

void Input()
{
    char number;

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}

    cout << "Image:" << endl;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << d[i][j];
		}
		cout << endl;
	}

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            inputLayer[pos]->outValue = d[i][j];
        }
	}

    label.read(&number, sizeof(char));
    for (int i = 0; i < n3; i++)
    {
        outputLayer[i]->expected = 0.0;
    }
    outputLayer[number]->expected = 1.0;

    cout << "Label: " << (int)(number) << endl;
}

void WriteMatrix(string file_name)
{
    ofstream file(file_name.c_str(), ios::out);
    cout << "open ok" << endl;
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            file << inputLayer[i]->weight[j] << " ";
        }
        file << endl;
    }

    cout << "input ok" << endl;

    for (int layer = 0; layer < nl2; layer++)
    {
        for (int i = 0; i < n2; i++)
        {
            int tempNum;
            if (layer == (nl2-1))
                tempNum = n3;
            else
                tempNum = n2;
            for (int j = 0; j < tempNum; j++)
                file << hiddenLayer[layer][i]->weight[j] << " ";
        }
        file << endl;
    }
    file.close();
}

int main(void)
{
    report.open(report_fn.c_str(), ios::out);

    InitNN();

    cout << "Init end" << endl;
    for (int trainTimes = 0; trainTimes < nTrainingTimes; trainTimes++)
    {
        image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
        label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

        char number;
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }

        for (int sample = 0; sample < nTraining; ++sample)
        {
            cout << "trainTimes " << trainTimes+1 << endl;
            cout << "Sample " << sample+1 << endl;

            Input();

            //cout << "Input" << endl;
            int nIterations = learning_process();

            // Write down the squared error
            cout << "No. iterations: " << nIterations << endl;
            printf("Error: %0.6lf\n\n", square_error());
            report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;
            
            // Save the current network (weights)
            if (sample % 100 == 0) 
            {
                cout << "Saving the network to " << model_fn << " file." << endl;
                WriteMatrix(model_fn);
            }
        }
        image.close();
        label.close();
    }

    WriteMatrix(model_fn);
    report.close();

    return 0;
}