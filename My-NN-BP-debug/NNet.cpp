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
                    hiddenLayer[i][j]->wDeltaSum.push_back(0.f);
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
        error += tmpe * tmpe / 2;
        //cout << "error: " << error << endl;
        // 公式2，3
        outputLayer[i]->delta 
            = (outputLayer[i]->value-outputLayer[i]->rightout)*(1-outputLayer[i]->value)*outputLayer[i]->value;
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
                hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
            }
            else
            {
                // 公式4
                for (int k=0; k<hideNodeNum; k++){sum += hiddenLayer[i + 1][k]->delta * hiddenLayer[i][j]->weight[k];}
                hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
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
    unsigned long times = 0;
    int sampleNum = sampleGroup.size();
    double result = 0.f;
    cout << "Num:" << sampleNum << endl;

    while(error > threshold)
    {
        //cout << "training error: " << error << endl;
        error = 0.f;
        // initialize delta sum
        for (int i = 0; i < inputNodeNum; i++) inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);
        for (int i = 0; i < hideLayer; i++){
            for (int j = 0; j < hideNodeNum; j++) 
            {
                hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.f);
                hiddenLayer[i][j]->bDeltaSum = 0.f;
            }
        }
        for (int i = 0; i < outputNodeNum; i++) outputLayer[i]->bDeltaSum = 0.f;

        for (int iter = 0; iter < sampleNum; iter++)
        {
            setInput(sampleGroup[iter].in);
            setOutput(sampleGroup[iter].out);

            forwardPropagationEpoc();
            backPropagationEpoc();
        }

        updateWeight(sampleNum);
        times++;
        if ((times % 10) == 0)
        {
            result = predict(sampleGroup);
            cout << "训练次数: " << times << endl;
            cout << "测试正确率: " << result << endl;
            cout << "training error: " << error << endl;
        }
    }
    cout << endl;

    // cout << "误差率: " << error << endl;
    // cout << "训练次数：" << times << endl;
    // for (int i = 0; i < inputNodeNum; i++)
    // {
    //     for (int j = 0; j < hideNodeNum; j++) 
    //     {
    //         cout << "输入层第" << i << "个节点的权重: " << inputLayer[i]->weight[j] << endl;
    //     }
    // }

    // for (int i = 0; i < hideLayer; i++)
    // {
    //     if (i == hideLayer - 1)
    //     {
    //         for (int j = 0; j < hideNodeNum; j++)
    //         { 
    //             // bias
    //             cout << "隐藏层第" << i << "层" << j << "个节点的偏量: " << hiddenLayer[i][j]->bias << endl;
    //             // weight
    //             for (int k = 0; k < outputNodeNum; k++) 
    //             { cout << "隐藏层第" << i << "层" << j << "个节点对" << k << "的权重: " << hiddenLayer[i][j]->weight[k]<<endl;}
    //         }
    //     }
    //     else
    //     {
    //         for (int j = 0; j < hideNodeNum; j++)
    //         {
    //             // bias
    //             cout << "隐藏层第" << i << "层" << j << "个节点的偏量: " << hiddenLayer[i][j]->bias << endl;
    //             // weight
    //             for (int k = 0; k < hideNodeNum; k++) 
    //             { cout << "隐藏层第" << i << "层" << j << "个节点对" << k << "的权重: " << hiddenLayer[i][j]->weight[k]<<endl; }
    //         }
    //     }
    //     for (int i = 0; i < outputNodeNum; i++)
    //     {  cout << "输出层第" << i << "个节点的偏量: " << outputLayer[i]->bias << endl; }
    // }
    
}

double NNet::predict(vector<sample>& testGroup)
{
    int testNum = testGroup.size();
    double result = 0.f;
    double maxValue = 0.f;
    int label = 0;
    int rightCount = 0;
    vector<int> errorLabel;
    vector<int> errorIter;
    vector<int> errorNumber;

    cout << "testNum:" << testNum << endl;
    for (int iter = 0; iter < testNum; iter++)
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
                    //cout << "sum:" << sum << endl;
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
            outputLayer[i]->value = sigmoid(sum);
            //testGroup[iter].out.push_back(outputLayer[i]->value);
            //cout << "maxValue: " << outputLayer[i]->value << endl;
            if (outputLayer[i]->value > maxValue)
            {
                maxValue = outputLayer[i]->value;
                //cout << "maxValue: " << maxValue << endl;
                label = i;
            }
        }

        if (1.0 == testGroup[iter].out.at(label))
        {
            rightCount++;
           // cout << "label:" << label << endl;
        }
        else
        {
            for (int i = 0; i < (int)testGroup[iter].out.size(); i++)
            {
                if (testGroup[iter].out.at(i) == 1.0)
                {
                    errorLabel.push_back(i);
                }
            }
            errorIter.push_back(label);
            errorNumber.push_back(iter);
        }
    }
    result = (double)rightCount/testNum;
    //cout << "正确: " << rightCount << endl;
    //cout << "正确率: " << result << endl;
    cout << "错误的是： " << endl;
    for (int i = 0; i < (int)errorLabel.size(); i++)
    {
        cout << "第" << errorNumber.at(i) << "个样本为：";
        cout << errorLabel.at(i) << "识别为：";
        cout << errorIter.at(i) << endl;
    }
    cout << endl;

    errorLabel.clear();
    errorLabel.shrink_to_fit();
    errorIter.clear();
    errorIter.shrink_to_fit();
    errorNumber.clear();
    errorNumber.shrink_to_fit();
    return result;
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
    // backward propagation on input layer
    for (int i = 0; i < inputNodeNum; i++)
    {
        for (int j = 0; j < hideNodeNum; j++) 
        {
            inputLayer[i]->weight[j] -= learningRate * inputLayer[i]->wDeltaSum[j] / sampleNum;
        }
    }

    // backward propagation on hidden layer
    // -- update weight & bias
    for (int i = 0; i < hideLayer; i++)
    {
        if (i == hideLayer - 1)
        {
            for (int j = 0; j < hideNodeNum; j++)
            { 
                // bias
                hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

                // weight
                for (int k = 0; k < outputNodeNum; k++) 
                { hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum; }
            }
        }
        else
        {
            for (int j = 0; j < hideNodeNum; j++)
            {
                // bias
                hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

                // weight
                for (int k = 0; k < hideNodeNum; k++) 
                { hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum; }
            }
        }
    }

    // backward propagation on output layer
    // -- update bias
    for (int i = 0; i < outputNodeNum; i++)
    { outputLayer[i]->bias -= learningRate * outputLayer[i]->bDeltaSum / sampleNum; }
}