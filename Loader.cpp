#include "Loader.h"
#include <fstream>
#include <vector>
#include <math.h>

int iterations = 1500;
float alpha = 0.01;
int m;
MatrixXf sample;
Matrix<float, 2, 1> theta;
MatrixXf Y;
MatrixXf X;

//element wise product of matrices
MatrixXf dotProduct(MatrixXf  p, MatrixXf q) {
	MatrixXf pq(p.rows(), p.cols());
	
	if (p.rows() == q.rows() && p.cols() == q.cols()) {
		for (int i = 0; i < p.rows(); i++) {
			for (int j = 0; j < p.cols(); j++) {
				pq(i, j) = p(i, j) * q(i, j);
			}
		}
	}
	else {
		pq.setZero();
		cout << "Matrices Must be Of Same Dimensions" << endl << "P is " << p.rows() << "X" << p.cols()
			 << " While Q is " << q.rows() << "X" << q.cols() << endl;
	}

	return pq;
}

//gradient descent to obtain the minimal value of theta
void gradientDescent(string filename) {
	sample = Load(filename);
	X.setOnes(m, 2);
	Y.setOnes(m, 1);

	// add first column of sample as second column of X and first column of x, second column as first column of Y
	for (int i = 0; i < m; i++) {
		X(i, 1) = sample(i, 0);
		Y(i, 0) = sample(i, 1);
	}

	Matrix<float, 2, 1> temp;
	temp.setZero(); //initialize theta with zeros

	for (int e = 0; e < iterations; e++) {
		temp(0, 0) = temp(0, 0) - (alpha / m) * (X * theta - Y).sum();
		temp(1, 0) = temp(1, 0) - (alpha / m) * (dotProduct(X * theta - Y, X(all, 1))).sum();

		theta(0, 0) = temp(0, 0);
		theta(1, 0) = temp(1, 0);
	}
}

//use obtained values of theta to compute a prediction
float predict(float input, float scale) {
	cout << "Using Theat values theta0 " << theta(0, 0) << " theta1 " << theta(1, 0) << endl;
	return input * theta(1, 0) + theta(0, 0)*scale;
}

//obtain numerical values from single line of the training set file
vector<float> getElements(string input) {
	vector<float> values;

	string temp;
	char seperator = ' '; //replace with ',' to read a csv data set file

	for (int i = 0; i < input.length(); i++) {
		char ltr = input[i];
		if (ltr == seperator) {
			values.push_back((float)stod(temp));
			temp = "";
		}
		else {
			temp += ltr;
		}
	}

	//add the last value that isnt character terminated
	values.push_back((float)stod(temp));

	return values;
}

//function to load training set into matrix
Matrix<float, Dynamic, Dynamic> Load(string filename) {
	ifstream file(filename);
	int row = 0, col = 0;
	vector<vector<float>> elements;
	vector<float> rowElements;

	//determine the number of rows and columns from the file provided
	//number of rows = number of lines in the file = length of elements
	//number of columns = number of seperate chars in the first line = length of rowElements

	string input="";

	//get file data
	while (getline(file, input)) {
		rowElements = getElements(input);
		elements.push_back(rowElements);
	}

	//declare a dynamic matrix
	row = elements.size(), col = rowElements.size();
	MatrixXf data(row, col);

	//set value of m, m = row count
	m = row;
	
	for (int i = 0; i < row; i++) {
		rowElements = elements.at(i);
		for (int j = 0; j < col; j++) {
			data(i, j) = rowElements.at(j);
		}
	}
	
	file.close();

	return data;
}

