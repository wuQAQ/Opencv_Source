#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

#define inputNodeNum    25
#define hideNodeNum     7
#define hideLayer       2
#define outputNodeNum   10
//#define learningRate (0.9)

// --- -1~1 随机数产生器 --- 
inline double get_11Random()    // -1 ~ 1
{
    return ((2.0*(double)rand()/RAND_MAX) - 1);
}

// --- sigmoid 函数 --- 
inline double sigmoid(double x)
{
    double ans = 1 / (1+exp(-x));
    return ans;
}

inline double sigmoidDelta(double x)
{
    double ans = x / (1-x);
    return ans;
}

// --- 输入层节点。包含以下分量：--- 
// 1.value:     固定输入值； 
// 2.weight:    面对第一层隐含层每个节点都有权值； 
// 3.wDeltaSum: 面对第一层隐含层每个节点权值的偏导数值累积
typedef struct inputNode
{
    double value;
    vector<double> weight;
    vector<double> wDeltaSum;
}inputNode;

// --- 输出层节点。包含以下数值：--- 
// 1.value:     节点当前值； 
// 2.delta:     与正确输出值之间的delta值； 
// 3.rightout:  正确输出值
// 4.bias:      偏移量
// 5.bDeltaSum: bias的delta值的累积，每个节点一个
typedef struct outputNode   // 输出层节点
{
    double value, delta, rightout, bias, bDeltaSum;
}outputNode;

// --- 隐含层节点。包含以下数值：--- 
// 1.value:     节点当前值； 
// 2.delta:     BP推导出的delta值；
// 3.bias:      偏移量
// 4.bDeltaSum: bias的delta值的累积，每个节点一个
// 5.weight:    面对下一层（隐含层/输出层）每个节点都有权值； 
// 6.wDeltaSum： weight的delta值的累积，面对下一层（隐含层/输出层）每个节点各自积累
typedef struct hiddenNode   // 隐含层节点
{
    double value, delta, bias, bDeltaSum;
    vector<double> weight, wDeltaSum;
}hiddenNode;

// --- 单个样本 --- 
typedef struct sample
{
    vector<double> in, out;
}sample;

// --- BP神经网络 --- 
class NNet
{
public:
    NNet();
    void forwardPropagationEpoc();
    void backPropagationEpoc(); 

    void training (vector<sample> sampleGroup, double threshold);
    void predict  (vector<sample>& testGroup); 

    void setInput (vector<double> sampleIn);     
    void setOutput(vector<double> sampleOut);    

public:
    double error;
    inputNode* inputLayer[inputNodeNum];                      // 输入层（仅一层）
    outputNode* outputLayer[outputNodeNum];                   // 输出层（仅一层）
    hiddenNode* hiddenLayer[hideLayer][hideNodeNum];       // 隐含层（可能有多层）
};