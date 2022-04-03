#pragma once
#include "types.h"

class PCA {

public:

    PCA(unsigned int n_components, unsigned num_iter=5000, double epsilon=1e-16):
     _nComponents(n_components), _numIter(num_iter), _epsilon(epsilon), _mu(), _mCov(), _eigValues(), _eigVectors()
    {};

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);

    Matrix get_cov();

private:

    unsigned int _nComponents;
    unsigned _numIter;
    double _epsilon;

    Vector _mu;

    Matrix _mCov;
    Vector _eigValues;
    Matrix _eigVectors;
    

};
