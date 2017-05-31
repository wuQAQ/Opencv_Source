#include "NNet.h"

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
            if ((hideLayer - 1) == i)
            {
                for (int k = 0; k < outputNodeNum; k++) 
                {
                    hiddenLayer[i][j]->weight.push_back(get_11Random());
                }
            }
            else
            {
                for (int k = 0; k < hideNodeNum; k++) 
                {
                    hiddenLayer[i][j]->weight.push_back(get_11Random());
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

void NNet::forwardPropagationEpoc()
{
    // forward propagation on hidden layer
    for (int i = 0; i < hideLayer; i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < hideNodeNum; j++)
            {
                double sum = 0.f;
                for (int k = 0; k < inputNodeNum; k++) 
                {
                    sum += inputLayer[k]->value * inputLayer[k]->weight[j];
                }
                sum += hiddenLayer[i][j]->bias;
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
    }
}

void NNet::backPropagationEpoc()
{
    // backward propagation on output layer
    // -- compute delta
    for (int i = 0; i < outputNodeNum; i++)
    {
        // 计算costValue
        double tmpe = fabs(outputLayer[i]->value-outputLayer[i]->rightout);
        error += tmpe * tmpe / 2;

        outputLayer[i]->delta 
            = (outputLayer[i]->value-outputLayer[i]->rightout)*sigmoidDelta(outputLayer[i]->value);
    }

    // backward propagation on hidden layer
    // -- compute delta
    for (int i = hideLayer - 1; i >= 0; i--)    // 反向计算
    {
        for (int j = 0; j < hideNodeNum; j++)
        {
            double sum = 0.f;
            if ((hideLayer - 1) == i)
            {
                for (int k=0; k<outputNodeNum; k++){sum += outputLayer[k]->delta * hiddenLayer[i][j]->weight[k];}
                hiddenLayer[i][j]->delta = sum * sigmoidDelta(hiddenLayer[i][j]->value);
            }
            else
            {
                for (int k=0; k<hideNodeNum; k++){sum += hiddenLayer[i + 1][k]->delta * hiddenLayer[i][j]->weight[k];}
                hiddenLayer[i][j]->delta = sum * sigmoidDelta(hiddenLayer[i][j]->value);
            }
        }
    }

    // backward propagation on input layer
    // -- update weight delta sum
    for (int i = 0; i < inputNodeNum; i++)
    {
        for (int j = 0; j < hideNodeNum; j++)
        {
            inputLayer[i]->wDeltaSum[j] += inputLayer[i]->value * hiddenLayer[0][j]->delta;
        }
    }

    // backward propagation on hidden layer
    // -- update weight delta sum & bias delta sum
    for (int i = 0; i < hideLayer; i++)
    {
        if (i == hideLayer - 1)
        {
            for (int j = 0; j < hideNodeNum; j++)
            {
                hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
                for (int k = 0; k < outputNodeNum; k++)
                { hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * outputLayer[k]->delta; }
            }
        }
        else
        {
            for (int j = 0; j < hideNodeNum; j++)
            {
                hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
                for (int k = 0; k < hideNodeNum; k++)
                { hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * hiddenLayer[i+1][k]->delta; }
            }
        }
    }

    // backward propagation on output layer
    // -- update bias delta sum
    for (int i = 0; i < outputNodeNum; i++) outputLayer[i]->bDeltaSum += outputLayer[i]->delta;
}


void NNet::training(vector<sample> sampleGroup, double threshold)
{
    int sampleNum = sampleGroup.size();
    error = 100.f;
    while(error > threshold)
    {
        cout << "training error: " << error << endl;
        error = 0.f;
        // initialize delta sum
        for (int i = 0; i < inputNodeNum; i++) 
            inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);

        for (int i = 0; i < hideLayer; i++){
            for (int j = 0; j < inputNodeNum; j++) 
            {
                hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.f);
                hiddenLayer[i][j]->bDeltaSum = 0.f;
            }
        }

        for (int i = 0; i < outputNodeNum; i++) 
            outputLayer[i]->bDeltaSum = 0.f;

        for (int iter = 0; iter < sampleNum; iter++)
        {
            setInput(sampleGroup[iter].in);
            setOutput(sampleGroup[iter].out);

            forwardPropagationEpoc();
            backPropagationEpoc();
        }

        // backward propagation on input layer
        // -- update weight
        for (int i = 0; i < inputNodeNum; i++)
        {
            for (int j = 0; j < hideNodeNum; j++) 
            {
                inputLayer[i]->weight[j] -= 0.9 * inputLayer[i]->wDeltaSum[j] / sampleNum;
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
                    hiddenLayer[i][j]->bias -= 0.9 * hiddenLayer[i][j]->bDeltaSum / sampleNum;

                    // weight
                    for (int k = 0; k < outputNodeNum; k++) 
                    { hiddenLayer[i][j]->weight[k] -= 0.9 * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum; }
                }
            }
            else
            {
                for (int j = 0; j < hideNodeNum; j++)
                {
                    // bias
                    hiddenLayer[i][j]->bias -= 0.9 * hiddenLayer[i][j]->bDeltaSum / sampleNum;

                    // weight
                    for (int k = 0; k < hideNodeNum; k++) 
                    { hiddenLayer[i][j]->weight[k] -= 0.9 * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum; }
                }
            }
        }

        // backward propagation on output layer
        // -- update bias
        for (int i = 0; i < outputNodeNum; i++)
        { outputLayer[i]->bias -= 0.9 * outputLayer[i]->bDeltaSum / sampleNum; }
    }
}

void NNet::predict(vector<sample>& testGroup)
{
    int testNum = testGroup.size();

    for (int iter = 0; iter < testNum; iter++)
    {
        testGroup[iter].out.clear();
        setInput(testGroup[iter].in);

        // forward propagation on hidden layer
        for (int i = 0; i < hideLayer; i++)
        {
            if (i == 0)
            {
                for (int j = 0; j < hideNodeNum; j++)
                {
                    double sum = 0.f;
                    for (int k = 0; k < inputNodeNum; k++) 
                    {
                        sum += inputLayer[k]->value * inputLayer[k]->weight[j];
                    }
                    sum += hiddenLayer[i][j]->bias;
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
            testGroup[iter].out.push_back(outputLayer[i]->value);
        }
    }
}

void NNet::setInput(vector<double> sampleIn)
{
    for (int i = 0; i < inputNodeNum; i++) 
        inputLayer[i]->value = sampleIn[i];
}

void NNet::setOutput( vector<double> sampleOut)
{
    for (int i = 0; i < outputNodeNum; i++) 
        outputLayer[i]->rightout = sampleOut[i];
}