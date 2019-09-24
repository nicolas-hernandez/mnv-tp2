#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
    this->neighbors = n_neighbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
    this->training_samples = X;
    this->training_labels = y;
}


Vector KNNClassifier::predict(SparseMatrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        ret(k) = 0;
    }

    return ret;
}



Vector KNNClassifier::distance_to_row(Vector row)
{
    auto distances = Vector(this->training_samples.rows());
    int cols = this->training_samples.cols();
    SparseMatrix diffs;
    for(int i = 0; i < cols; ++i) {
          diffs.col(i) = this->training_samples.col(i) - row;
          distances(i) = diffs.col(i).norm();
    }
    return distances;
}
