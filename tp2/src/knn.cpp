#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <fstream>

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
  _n_neighbors = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y)
{
  training_data = X;
  target_values = y;
}


Vector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

      for (unsigned k = 0; k < ret.rows(); ++k)
      {
         ret(k) = single_predict(X.row(k));
      }
      
    return ret;

}

bool func_comp(const tuple<double, unsigned int> &izq, const tuple<double, unsigned int> &der){
  return (get<0>(izq) < get<0>(der));
}

int KNNClassifier::single_predict(Vector v){

  // calcular la distancia, label a todos los de training_data
  vector<tuple<long double, unsigned int> > distances;
  distances.reserve(training_data.rows());

for (unsigned int i = 0; i < training_data.rows(); i++) { // Recorro todas las imagenes del set comparandolas con mi imagen
    

    Vector resta = (training_data.row(i) - v.transpose());

    
    distances.push_back(make_tuple(resta.squaredNorm(), i));

}


  // ordeno por distancia y me quedo con los primeros k
  partial_sort(distances.begin(), distances.begin() + _n_neighbors, distances.end(), func_comp);
  // busco el label que mas se repite

  vector<unsigned int> appearances(10, 0);
  unsigned int label = 0;
  for(unsigned i = 0; i < _n_neighbors; i++){
    label = target_values(get<1>(distances[i]), 0);
    appearances[label]++;
  }

  unsigned int max_label = 0;
  unsigned int max_appearances = 0;

  for (unsigned int i = 0; i < 10; i++) {
    if(appearances[i] > max_appearances){
      max_label = i;
      max_appearances = appearances[i];
    }
  }

  return max_label;
}
