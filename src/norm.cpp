#include "norm.hpp"

double min_max_norm(const double x,
                    const double x_max,
                    const double x_min){
  // compute min_max norm
  return (x - x_min) / (x_max - x_min);
}
