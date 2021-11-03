#pragma once

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

Matrix<float, Dynamic, Dynamic> Load(string filename);

float predict(float input, float scale);

void gradientDescent(string filename);

MatrixXf dotProduct(MatrixXf  p, MatrixXf q);