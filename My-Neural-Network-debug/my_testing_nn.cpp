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

// Testing image file name
const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";

// Testing label file name
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "output/testing-report-28.dat";

// Number of testing samples
const int nTesting = 10000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

const int n1 = width * height; 
const int n2 = 128;
const int nl2 = 1;
const int n3 = 10; 

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
int d[width][height];

ifstream image;
ifstream label;
ofstream report;

void InitNN(void)
{
    // 1.分配空间
    for (int i = 0; i < n1; i++)
    {
        inputLayer[i] = new inputNode();
        for(int j = 0; j < n2; j++)
        {
            inputLayer[i]->weight.push_back(0.0);
        }
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

    for (int layer = 0; layer < nl2; layer++)
    {
        if (layer == 0)
        {
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {   
                    hiddenLayer[layer][j]->inValue += 
                        inputLayer[i]->outValue * inputLayer[i]->weight[j];
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
                        hiddenLayer[layer-1][i]->outValue * hiddenLayer[layer-1][i]->weight[j];
                }
            }
        }

        for (int i = 0; i < n2; i++)
        {
            int inValueTemp = hiddenLayer[layer][i]->inValue;
            hiddenLayer[layer][i]->outValue = sigmoid(inValueTemp);
        }
    }

    for (int i = 0; i < n2; i++)
    {
        for (int j = 0; j < n3; j++)
        {
            outputLayer[j]->inValue = hiddenLayer[nl2-1][i]->outValue * hiddenLayer[nl2-1][i]->weight[j];
        }
    }

    for (int i = 0; i < n3; i++)
    {
        int inValueTemp = outputLayer[i]->inValue;
        outputLayer[i]->outValue = sigmoid(inValueTemp);
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

int Input()
{
    char number;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int pos = i + j * width;
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
    return (int)(number);
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

void LoadModel(string file_name)
{
    ifstream file(file_name.c_str(), ios::in);

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            double temp = 0.0;
            file >> temp;
            inputLayer[i]->weight[j] = temp;
        }
    }

    for (int layer = 0; layer < nl2; layer++)
    {
        for (int i = 0; i < n2; i++)
        {
            int tempNum;
            if (layer == (nl2 - 1))
                tempNum = n3;
            else
                tempNum = n2;

            for (int j = 0; j < tempNum; j++)
            {
                double temp;
                file >> temp;
                hiddenLayer[layer][i]->weight.push_back(temp);
            }
        }
    }

    file.close();
}

int main(void)
{
    report.open(report_fn.c_str(), ios::out);
    image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}

    InitNN();
    
    LoadModel(model_fn);

    int nCorrect = 0;

    for (int sample = 0; sample < nTesting; ++sample)
    {
        cout << "Sample " << sample << endl;

        int label = Input();

        Perceptron();

        int predict = 0;
        for (int i = 1; i < n3; ++i)
        {
            cout << outputLayer[i]->outValue << endl;
            if (outputLayer[i]->outValue > outputLayer[predict]->expected)
            {
                predict = i;
            } 
        }

        double error = square_error();
        printf("Error: %0.6lf\n", error);

        if (label == predict) 
        {
           ++nCorrect;
			cout << "Classification: YES. Label = " << label << ". Predict = " << predict << endl << endl;
			report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		} 
        else 
        {
			cout << "Classification: NO.  Label = " << label << ". Predict = " << predict << endl;
			cout << "Image:" << endl;
			for (int j = 0; j < height; ++j) {
				for (int i = 0; i < width; ++i) {
					cout << d[i][j];
				}
				cout << endl;
			}
			cout << endl;
			report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
		}
    }

    // Summary
    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    printf("Accuracy: %0.2lf\n", accuracy);
    
    report << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();
    
    return 0;
}