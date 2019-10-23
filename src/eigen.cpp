#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    double eigenvalue;
    /***********************
     * COMPLETAR CODIGO
     **********************/
    double norma= 0;
    double normaActual = b.norm();
    b.normalize();
    int iter=0;

    while(abs(norma-normaActual<eps || iter <num_iter)){
        b = X*b;

        norma = normaActual;
        normaActual=b.norm();
        b.normalize();
        iter++;

    }
    eigenvalue= b.transpose()*X*b;

    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    /***********************
     * COMPLETAR CODIGO
     **********************/
    return make_pair(eigvalues, eigvectors);
}
