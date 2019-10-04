#include <iostream>
#include "pca.h"
#include "eigen.h"

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

  this->covariance_matrix = (1/(X.cols()-1)) * (X.transpose() * X);
  Eigen::EigenSolver<MatrixXd> es(this->covariance_matrix);
  //Eigen::MatrixXd eigenvectorsf = es.eigenvectors().real();
  this->eigenvectors = es.eigenvectors().real();
}


MatrixXd PCA::transform(SparseMatrix X)
{
  MatrixXd reduction = this->eigenvectors.block(0,0,this->eigenvectors.rows(),this->components);
  return X * reduction;
}
