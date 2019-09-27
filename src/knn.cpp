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
    // vector<bool> ret(X.rows());
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        ret(k) = this->predict_row(X.row(k));
    }

    return ret;
}


// Calculates the distances from every row of the training set
// to the row received as an argument
Vector KNNClassifier::distance_to_row(Vector row)
{ 
    auto distances = Vector(this->training_samples.rows());
    int rows = this->training_samples.rows();
    int cols = this->training_samples.cols();
    SparseMatrix diffs(rows, cols);
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; j++)
        { //TODO: hacerlo funcionar como resta de filas
            diffs.coeffRef(i, j) = this->training_samples.coeff(i, j) - row(j);
        }
        distances(i) = diffs.row(i).norm();
    }
    return distances;
}
//TODO: replace std vectors with eigen ones
bool KNNClassifier::predict_row(Vector row)
{
    Vector distances = this->distance_to_row(row);

    // index = np.argsort(dist)
    // from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    std::vector<int> closestIndex(distances.size());
    std::iota(closestIndex.begin(), closestIndex.end(), 0);

    std::sort(closestIndex.begin(),
              closestIndex.end(),
              [&distances](size_t i1, size_t i2) {return distances[i1] < distances[i2];});

    // closest = index[0:self.n_neighbors]
    // neighbors = [self.y[i] for i in closest]
    std::vector<bool> closest_neighbors(this->neighbors);
    for(int i = 0; i < this->neighbors; i++)
    {
        int index = closestIndex[i];
        closest_neighbors[i] = this->training_labels(index);
    }


    //count = np.bincount(neighbors)
    //ret = np.argmax(count)
    bool prediction = vote_popular(closest_neighbors);

    return prediction;
}

bool KNNClassifier::vote_popular(std::vector<bool> closest_neighbors)
{ 
    unsigned int positive_reviews = std::count(closest_neighbors.begin(), closest_neighbors.end(), true);
    return (positive_reviews > (closest_neighbors.size()/2));
}
