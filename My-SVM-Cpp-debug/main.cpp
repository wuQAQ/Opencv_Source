#include "svm.h"
#include <string.h>
#include<iostream>

using namespace std;

svm_parameter param;

void init_param(){
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

int main(){
	init_param();
	svm_problem prob;
	prob.l = 4;
	prob.y = new double[prob.l];
	double d;
	int probfeature = 2;

	if(param.gamma == 0) param.gamma = 0.5;
	svm_node *x_space = new svm_node[(probfeature+1)*prob.l];
	prob.x = new svm_node * [prob.l]; 
	cout<<"size: "<<sizeof(x_space)<<endl;

	x_space[0].index = 1;
	x_space[0].value = 190;
	x_space[1].index = 2;
	x_space[1].value = 70;
	x_space[2].index = -1;
	prob.x[0] = &x_space[0];
	prob.y[0] = 1;
	x_space[3].index = 1;
	x_space[3].value = 180;
	x_space[4].index = 2;
	x_space[4].value = 80;
	x_space[5].index = -1;
	prob.x[1] = &x_space[3];
	prob.y[1] = 1;
	x_space[6].index = 1;
	x_space[6].value = 161;
	x_space[7].index = 2;
	x_space[7].value = 45;
	x_space[8].index = -1;
	prob.x[2] = &x_space[6];
	prob.y[2] = -1;
	x_space[9].index = 1;
	x_space[9].value = 163;
	x_space[10].index = 2;
	x_space[10].value = 47;
	x_space[11].index = -1;
	prob.x[3] = &x_space[9];
	prob.y[3] = -1;
	svm_model *model = svm_train(&prob, &param);
	//predict ���180cm, ����85kg
	svm_node xnode[3];
	xnode[0].index = 1;
	xnode[0].value = 161;
	xnode[1].index = 2;
	xnode[1].value = 85;
	xnode[2].index = -1;
	d = svm_predict(model, xnode);
	cout<<d<<" "<<endl;
	//predict ���161cm,����50kg
	xnode[0].index = 1;
	xnode[0].value = 180;
	xnode[1].index = 2;
	xnode[1].value = 40;
	xnode[2].index = -1;
	d = svm_predict(model, xnode);
	cout<<d<<" "<<endl;

	delete[] x_space;
	delete[] prob.x;
	delete[] prob.y;
}