#include <iostream>
#include <iomanip>
#include "pca.h"
#include "eigen.h"
#include <queue>
#include "pm.hpp"

using namespace std;

PCA::PCA(unsigned int n_components)
{
	this->components = n_components;
}

void PCA::fit(Matrix X)
{
	int rows = X.rows();
	for(int i = 0; i < X.cols(); i++) {
		// calculo la mediana
		int mean = X.col(i).mean();

		for(int j = 0; j < rows; j++) {
			X(j, i) = (double)(X(j, i) - mean)/sqrt(rows-1);
		}
	}

	Matrix covarianza = X.transpose()*X/(rows-1);
	cout << "Cov matrix built\n";

	pair<Vector, Matrix> components = get_first_eigenvalues(covarianza,this->components,1000,0.01);
	cout << "Eigenvectors calculated\n";
	
	this->reduction = components.second;
}


MatrixXd PCA::transform(SparseMatrix X)
{
	cout << "Transforming\n";
	Matrix A(X);
	return this->reduction*A.transpose();
}

