#include "NNet.h"
#include <random>
#include <algorithm> 

using namespace std;

NNet::NNet()
{
    srand((unsigned)time(NULL));        // 随机数种子    
    error = 100.f;                      // error初始值，极大值即可

    // 初始化输入层
    for (int i = 0; i < inputNodeNum; i++)
    {
        inputLayer[i] = new inputNode();
        for (int j = 0; j < hideNodeNum; j++) 
        {
            inputLayer[i]->weight.push_back(get_11Random());
            inputLayer[i]->wDeltaSum.push_back(0.f);
        }
    }

    // 初始化隐藏层
    for (int i = 0; i < hideLayer; i++)
    {
        for (int j = 0; j < hideNodeNum; j++)
        {
            hiddenLayer[i][j] = new hiddenNode();
            hiddenLayer[i][j]->bias = get_11Random();
            // 如果是隐藏层的最后一层
            if ((hideLayer - 1) == i)
            {
                for (int k = 0; k < outputNodeNum; k++) 
                {
                    hiddenLayer[i][j]->weight.push_back(get_11Random());
                    hiddenLayer[i][j]->wDeltaSum.push_back(0.f);
                }
            }
            else
            {
                for (int k = 0; k < hideNodeNum; k++) 
                {
                    hiddenLayer[i][j]->weight.push_back(get_11Random());
                    //hiddenLayer[i][j]->wDeltaSum.push_back(0.f);
                }
            }
            
        }      
    }

    // 初始化输出层
    for (int i = 0; i < outputNodeNum; i++)
    {
        outputLayer[i] = new outputNode();
        outputLayer[i]->bias = get_11Random();
    }
}

// 前向传播
void NNet::forwardPropagationEpoc()
{
    // forward propagation on hidden layer
    for (int i = 0; i < hideLayer; i++)
    {
        for (int j = 0; j < hideNodeNum; j++)
        {
            double sum = 0.f;
            if (i == 0)
            {
                for (int k = 0; k < inputNodeNum; k++) 
                    sum += inputLayer[k]->value * inputLayer[k]->weight[j];
            }
            else
            {
                for (int k = 0; k < hideNodeNum; k++) 
                    sum += hiddenLayer[i-1][k]->value * hiddenLayer[i-1][k]->weight[j];
            }
            sum += hiddenLayer[i][j]->bias;
            hiddenLayer[i][j]->value = sigmoid(sum);
        }
    }

    // forward propagation on output layer
    for (int i = 0; i < outputNodeNum; i++)
    {
        double sum = 0.f;
        for (int j = 0; j < hideNodeNum; j++)
        {
            sum += hiddenLayer[hideLayer-1][j]->value * hiddenLayer[hideLayer-1][j]->weight[i];
        }
        sum += outputLayer[i]->bias;
        outputLayer[i]->value = sigmoid(sum);
    }
}

// 反向传播
void NNet::backPropagationEpoc()
{
    // backward propagation on output layer
    // -- compute delta
    for (int i = 0; i < outputNodeNum; i++)
    {
        // 计算训练误差值
        // 公式1
        double tmpe = fabs(outputLayer[i]->value-outputLayer[i]->rightout);
        if (isnan(tmpe))
            tmpe = 0.0001;
        error += tmpe * tmpe / 2;
        //cout << "error: " << error << endl;
        // 公式2，3
        outputLayer[i]->delta = (outputLayer[i]->value-outputLayer[i]->rightout)*sigmoidDerivatives(outputLayer[i]->value);
        //cout << "outputLayer[i]->delta: " << outputLayer[i]->delta << endl;
    }

    // cout << "hidden layer" << endl;
    // backward propagation on hidden layer
    // -- compute delta
    for (int i = hideLayer - 1; i >= 0; i--)    // 反向计算
    {
        for (int j = 0; j < hideNodeNum; j++)
        {
            double sum = 0.f;
            if ((hideLayer - 1) == i)
            {
                // 公式4
                for (int k=0; k<outputNodeNum; k++){sum += outputLayer[k]->delta * hiddenLayer[i][j]->weight[k];}
                hiddenLayer[i][j]->delta = sum * sigmoidDerivatives(hiddenLayer[i][j]->value);
            }
            else
            {
                // 公式4
                for (int k=0; k<hideNodeNum; k++){sum += hiddenLayer[i + 1][k]->delta * hiddenLayer[i][j]->weight[k];}
                hiddenLayer[i][j]->delta = sum * sigmoidDerivatives(hiddenLayer[i][j]->value);
            }
        }
    }

    // cout << "input layer" << endl;
    // backward propagation on input layer
    // -- update weight delta sum
    for (int i = 0; i < inputNodeNum; i++)
    {
        for (int j = 0; j < hideNodeNum; j++)
        {
            inputLayer[i]->wDeltaSum[j] += hiddenLayer[0][j]->delta * inputLayer[i]->value;
        }
    }

    // cout << "backward propagation on hidden layer" << endl;
    // backward propagation on hidden layer
    // -- update weight delta sum & bias delta sum
    for (int i = 0; i < hideLayer; i++)
    {
        for (int j = 0; j < hideNodeNum; j++)
        {
            if ((hideLayer-1) == i)
            {
                hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
                for (int k = 0; k < outputNodeNum; k++)
                { hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * outputLayer[k]->delta; }
            }
            else
            {
                hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
                // cout << "2hiddenLayer[i][j]->bDeltaSum: " << hiddenLayer[i][j]->bDeltaSum << endl;
                for (int k = 0; k < hideNodeNum; k++)
                { 
                    // cout << "K: " << k << endl;
                    // cout << "hiddenLayer[i][j]->value:" << hiddenLayer[i][j]->value << endl;
                    // cout << "hiddenLayer[i+1][k]->delta: " << hiddenLayer[i+1][k]->delta << endl;
                    // cout << "hiddenLayer[i][j]->wDeltaSum[k]: " << hiddenLayer[i][j]->wDeltaSum[k] << endl;
                    hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * hiddenLayer[i+1][k]->delta; 
               }
            }
        }
    }

    //cout << "backward propagation on output layer" << endl;
    // backward propagation on output layer
    // -- update bias delta sum
    for (int i = 0; i < outputNodeNum; i++) 
        outputLayer[i]->bDeltaSum += outputLayer[i]->delta;
}


void NNet::training(vector<sample> sampleGroup, double threshold)
{
    // 样本数量
    int sampleNum = sampleGroup.size();
    
    cout << "sampleNum: " << sampleNum << endl;
    random_shuffle (sampleGroup.begin(), sampleGroup.end());
    while(error > threshold)
    {
        cout << "training error: " << error << endl;
        error = 0.f;
        for (int i = 0; i < inputNodeNum; i++) 
            inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);

        // 隐藏层初始化权重为0，偏移为0
        for (int i = 0; i < hideLayer; i++)
        {
            for (int j = 0; j < hideNodeNum; j++) 
            {
                // cout << "hiddenLayer[i][j]->wDeltaSum.size() " <<hiddenLayer[i][j]->wDeltaSum.size() << endl;
                hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.f);
                hiddenLayer[i][j]->bDeltaSum = 0.f;
            }
        }

        // 输出初始化权重为0
        for (int i = 0; i < outputNodeNum; i++) 
            outputLayer[i]->bDeltaSum = 0.f;

        for (int iter = 0; iter < sampleNum; iter++)
        {
            setInput(sampleGroup[iter].in);
            setOutput(sampleGroup[iter].out);

            forwardPropagationEpoc();
            backPropagationEpoc();
        }

        updateWeight(sampleNum);
    }
}

void NNet::predict(vector<sample>& testGroup)
{
    int testNum = testGroup.size();
    double maxValue = 0.f;
    int label = 0;
    int rightCount = 0;
    cout << "testNum:" << testNum << endl;
    for (int iter = 0; iter < 1; iter++)
    {
        //testGroup[iter].out.clear();
        setInput(testGroup[iter].in);

        // forward propagation on hidden layer
        for (int i = 0; i < hideLayer; i++)
        {
            if (i == 0)
            {
                for (int j = 0; j < hideNodeNum; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < inputNodeNum; k++) 
                    {
                        sum += inputLayer[k]->value * inputLayer[k]->weight[j];
                    }
                    sum += hiddenLayer[i][j]->bias;
                    cout << "sum:" << sum << endl;
                    hiddenLayer[i][j]->value = sigmoid(sum);
                }
            }
            else
            {
                for (int j = 0; j < hideNodeNum; j++)
                {
                    double sum = 0.f;
                    for (int k = 0; k < hideNodeNum; k++) 
                    {
                        sum += hiddenLayer[i-1][k]->value * hiddenLayer[i-1][k]->weight[j];
                    }
                    sum += hiddenLayer[i][j]->bias;
                    hiddenLayer[i][j]->value = sigmoid(sum);
                }
            }
        }

        // forward propagation on output layer
        for (int i = 0; i < outputNodeNum; i++)
        {
            double sum = 0.f;
            for (int j = 0; j < hideNodeNum; j++)
            {
                sum += hiddenLayer[hideLayer-1][j]->value * hiddenLayer[hideLayer-1][j]->weight[i];
            }
            sum += outputLayer[i]->bias;
            cout << sum << endl;
            outputLayer[i]->value = sigmoid(sum);
            //testGroup[iter].out.push_back(outputLayer[i]->value);
            cout << "maxValue: " << outputLayer[i]->value << endl;
            if (outputLayer[i]->value > maxValue)
            {
                maxValue = outputLayer[i]->value;
                cout << "maxValue: " << maxValue << endl;
                label = i;
            }
        }

        if (1.0 == testGroup[iter].out.at(label))
        {
            rightCount++;
            cout << label << endl;
        }
    }
    cout << "正确率: " << (double)rightCount/testNum << endl;
}

void NNet::setInput(vector<double> sampleIn)
{
    for (int i = 0; i < inputNodeNum; i++) 
        inputLayer[i]->value = sampleIn.at(i);
}

void NNet::setOutput(vector<double> sampleOut)
{
    for (int i = 0; i < outputNodeNum; i++) 
        outputLayer[i]->rightout = sampleOut.at(i);
}

void NNet::updateWeight(int sampleNum)
{
    for (int i = 0; i < inputNodeNum; i++)
    {
        for (int j = 0; j < hideNodeNum; j++) 
        {
            inputLayer[i]->weight[j] -= (0.9 * inputLayer[i]->wDeltaSum[j] / sampleNum);
        }
    }

    for (int i = 0; i < hideLayer; i++)
    {
        if (i == hideLayer - 1)
        {
            for (int j = 0; j < hideNodeNum; j++)
            { 
                // bias
                hiddenLayer[i][j]->bias -= (0.9 * hiddenLayer[i][j]->bDeltaSum / sampleNum);

                // weight
                for (int k = 0; k < outputNodeNum; k++) 
                { hiddenLayer[i][j]->weight[k] -= (0.9 * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum); }
            }
        }
        else
        {
            for (int j = 0; j < hideNodeNum; j++)
            {
                // bias
                hiddenLayer[i][j]->bias -= (0.9 * hiddenLayer[i][j]->bDeltaSum / sampleNum);

                // weight
                for (int k = 0; k < hideNodeNum; k++) 
                { hiddenLayer[i][j]->weight[k] -= (0.9 * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum); }
            }
        }
    }

    for (int i = 0; i < outputNodeNum; i++)
    { 
        outputLayer[i]->bias -= (0.9 * outputLayer[i]->bDeltaSum / sampleNum); 
    }
}