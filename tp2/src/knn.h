#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

    int single_predict(Vector v);
private:
    
  unsigned int _n_neighbors;
  Matrix training_data;
  Matrix target_values;
};
