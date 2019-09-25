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
        ret(k) = this->predict_row(X.row(k));
    }

    return ret;
}



Vector KNNClassifier::distance_to_row(Vector row)
{
    auto distances = Vector(this->training_samples.rows());
    int rows = this->training_samples.rows();
    int cols = this->training_samples.cols();
    SparseMatrix diffs(rows, cols);
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; j++) { //TODO: hacerlo funcionar como resta de filas
            diffs.coeffRef(i, j) = this->training_samples.coeff(i, j) - row(j);
        }
        distances(i) = diffs.row(i).norm();
    }
    return distances;
}

double KNNClassifier::predict_row(Vector row)
{
    Vector distances = this->distance_to_row(row);
    
    //index = np.argsort(dist)
    //closest = index[0:self.n_neighbors]
    //neighbors = [self.y[i] for i in closest]
    //count = np.bincount(neighbors)
    //ret = np.argmax(count)
	
	return 0.0;
}
