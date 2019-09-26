#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(SparseMatrix X, Matrix y);

    Vector predict(SparseMatrix X);
private:
    Vector distance_to_row(Vector row);
    double vote_popular(std::vector<double> closest_neighbors);
    double predict_row(Vector row);
    int neighbors;
    SparseMatrix training_samples;
    Matrix training_labels;
};
