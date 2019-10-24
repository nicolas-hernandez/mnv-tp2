#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"
#include <cmath>
using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    double eigenvalue;
    double norma= 0;
    double normaActual = b.norm();
    b.normalize();
    int iter=0;

    while(abs(norma-normaActual)<eps || iter <num_iter){
        b = X*b;

        norma = normaActual;
        normaActual=b.norm();
        b.normalize();
        iter++;
    }

    eigenvalue= (X*b).norm() ;

    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);
    for(int i = 0; i < num; i++){
        pair<double, Vector> av = power_iteration(A, num_iter, epsilon);
        double lambda = av.first;
        Vector v = av.second;
        eigvectors.row(i) = v;
        eigvalues(i) = lambda;
        A = A - lambda*v*v.transpose();
    }

    return make_pair(eigvalues, eigvectors);
}
