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
	for(int i = 0; i < X.cols(); i++) {
		int n = X.rows();
		Vector meanVector(n);

		// calculo la mediana
		int mean = X.row(i).mean();

		for(int j = 0; j < X.rows(); j++) {
			X(i, j) = (double)(X(i, j) - mean)/sqrt(n-1);
		}
	}

	Matrix covarianza = X.transpose()*X/(X.rows()-1);

	pair<Vector, Matrix> components = get_first_eigenvalues(X,this->components,10000,0.000001);
	this->reduction = components.second.transpose();
}


MatrixXd PCA::transform(SparseMatrix X)
{
	Matrix A(X);
	return this->reduction*A.transpose();
}

