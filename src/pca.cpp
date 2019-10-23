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

	Matrix covarianza = X.transpose()*X;

	auto comp = [](pair<Matrix, double> aV1, pair<Matrix, double> aV2) {
		return aV1.second > aV2.second;
	};

	priority_queue<pair<Matrix, double>, vector<pair<Matrix, double> >, decltype(comp)> queue(comp);

	std::vector<Vector> autovectores;

	int dim= covarianza.rows();

	for (int i = 0; i < this->components; i++) {
        
        pair autov = power_iteration(X, 100000, 0.0000001)

		autovectores.push_back(autovector);

		//busco autovalor
		Vector aux = covarianza*autovector;
		double lambda = aux.norm();

		Vector autovectorAux(autovector);
		autovector = autovector * lambda;
		covarianza = covarianza - (autovector.transpose() * autovectorAux);

		pair<Matrix, double> av = make_pair(autovector, lambda);
		queue.push(av);
	}

	this->reduction = Matrix(autovectores.size(), this->components);
	for(int ri = 0; ri < this->components; ri++) {
		this->reduction.col(ri) = queue.top().first;
		queue.pop();
	}

}


MatrixXd PCA::transform(SparseMatrix X)
{
	//MatrixXd reduction = this->eigenvectors.block(0,0,this->eigenvectors.rows(),this->components);
	return X * this->reduction;
}

