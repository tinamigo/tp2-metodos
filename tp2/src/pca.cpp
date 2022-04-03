#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


// PCA::PCA(unsigned int n_components, unsigned num_iter=5000, double epsilon=1e-16) :
// _nComponents(n_components), _numIter(num_iter), _epsilon(epsilon)
// {
// }


Matrix PCA::get_cov() {
  return this->_mCov;
}

void PCA::fit(Matrix X)
{
  // Obtenemos matriz de covarianza _mCov.
  this->_mu = X.colwise().mean();
  MatrixXd M = X.rowwise() - this->_mu.transpose();
  this->_mCov = (M.transpose() * M) / sqrt(X.rows()-1);

  // Obtenemos autovalores y autovectores de _mCov, _eigValues y _eigVectors.
  pair<Vector, Matrix> eigens = get_first_eigenvalues(this->_mCov, this->_nComponents,
                                                      this->_numIter, this->_epsilon);
  this->_eigValues = eigens.first;
  this->_eigVectors = eigens.second;
}


MatrixXd PCA::transform(Matrix X)
{
  // Aplicamos la transformación X'^T = V^T * X^T, siendo V la matriz con los autovectores de _mCov (_eigVectors).
  //  X'^T = V^T * X^T 

  MatrixXd X_T = (this->_eigVectors).transpose() * X.transpose();
  return X_T.transpose();
}
