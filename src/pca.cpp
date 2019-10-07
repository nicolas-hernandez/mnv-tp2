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
  cout  << X <<"\n";
  for(int c = 0; c < X.cols(); c++) {
    Vector meanVector(X.rows());
    int mean = X.col(c).mean();
    for(int i = 0; i < X.rows(); i++) {
      meanVector[i] = mean;
    }
    X.col(c) = X.col(c) - meanVector;
  }

  cout << X <<"\n";
  Matrix covarianza = (1/(X.cols()-1)) * (X.transpose() * X);
  cout << std::setprecision(5)<< covarianza << "\n";

  auto comp = [](pair<Matrix, double> aV1, pair<Matrix, double> aV2) {
    return aV1.second > aV2.second;
  };
  priority_queue<pair<Matrix, double>, vector<pair<Matrix, double> >, decltype(comp)> queue(comp);
  std::vector<Vector> autovectores;
  
  for (int i = 0; i < this->components; i++) {
	Vector autovector = metodoDeLasPotencias(covarianza);
	autovectores.push_back(autovector);
	double lamda = encontrarAutovalor(autovector, covarianza);
	Vector autovectorAux(autovector);
	autovector = autovector * lamda;
	covarianza = covarianza - (autovector.transpose() * autovectorAux);
    
	pair<Matrix, double> av = make_pair(autovector, lamda);
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

