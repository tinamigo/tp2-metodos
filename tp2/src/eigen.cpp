#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    double eigenvalue;
    Vector z=Vector::Zero(X.cols());
    for(unsigned k=0;k<num_iter && !(b - z).isZero(eps);k++){
        z=b;
        b=X*b;
        b=b/ b.norm();
    }
    eigenvalue=z.transpose().dot(X*z)/z.norm();

    return make_pair(eigenvalue, b);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);
    for(unsigned i=0;i<num;i++){
        pair<double, Vector> resActual=power_iteration(A,num_iter,epsilon);
        eigvalues(i)=resActual.first;
        eigvectors.col(i)=resActual.second;
        A = A - resActual.first * resActual.second * resActual.second.transpose();
    }
    return make_pair(eigvalues, eigvectors);
}
