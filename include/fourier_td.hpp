#ifndef __FOURIER__TD
#define __FOURIER__TD
#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>

#include "conventions.hpp"


/*
 * TD algorithm
 */
Eigen::VectorXd
temporal_difference(const int num_iterations, // number of iterations to update
                         const int weightSize, // weight vector length (N,)
                         // problem specific tdUpdate function, returns when episode ends
                         void (*tdFunc)(Eigen::VectorXd& weights)
                         );

void save_file(const std::string filename);
#endif
