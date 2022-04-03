#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

// typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
using Eigen::MatrixXd;

// rowMajor: los valores de la matriz(doubles) son guardados en el orden de las filas en memoria.
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

// typedef Vector<double, Dynamic> VectorXd;
typedef Eigen::VectorXd Vector;
