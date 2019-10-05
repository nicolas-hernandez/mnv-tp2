#include <iostream>
#include "pca.h"
#include "eigen.h"
#include <queue>

using namespace std;


PCA::PCA(unsigned int n_components)
{
  this->components = n_components;
}

void PCA::fit(Matrix X)
{
  for(int c = 0; c < X.cols(); c++) {
    Vector meanVector(X.rows());
    int mean = X.col(c).mean();
    for(int i = 0; i < X.rows(); i++) {
      meanVector[i] = mean;
    }
    X.col(c) = X.col(c) - meanVector;
  }

  Matrix covariance_matrix = (1/(X.cols()-1)) * (X.transpose() * X);
  Eigen::EigenSolver<MatrixXd> es(covariance_matrix);
  auto comp = [](pair<Matrix, double> aV1, pair<Matrix, double> aV2) {
    return aV1.second > aV2.second;
  };
  
  priority_queue<pair<Matrix, double>, vector<pair<Matrix, double> >, decltype(comp)> queue(comp);
  Matrix eigenvectors = es.eigenvectors().real();
  Vector eigenvalues = es.eigenvalues().real();
  
  for(int ev = 0; ev < eigenvectors.cols(); ev++) {
    pair<Matrix, double> aV = make_pair(eigenvectors.col(ev),eigenvalues[ev]);
    queue.push(aV);
  }
  this->reduction = Matrix(eigenvectors.rows(), this->components);
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
